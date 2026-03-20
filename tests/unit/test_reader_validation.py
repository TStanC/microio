from __future__ import annotations

import json
from pathlib import Path
import shutil
import uuid

from microio.reader.open import open_dataset


def test_lif_scenes_do_not_auto_repair_t(lif_subset):
    ds = open_dataset(lif_subset, mode="a")
    repair = ds.repair_axis_metadata("14", persist=True)
    assert repair.persisted is True
    assert any(message.code in {"t_not_repaired", "t_unresolved"} for message in repair.warnings)
    assert repair.axis_states["t"].placeholder is True


def test_placeholder_xy_is_hard_error():
    dataset = Path(__file__).resolve().parents[3] / "data_out" / f"xy_placeholder_{uuid.uuid4().hex}.zarr"
    if dataset.exists():
        shutil.rmtree(dataset, ignore_errors=True)
    dataset.mkdir()
    (dataset / ".zgroup").write_text('{"zarr_format": 2}')
    (dataset / ".zattrs").write_text('{"bioformats2raw.layout": 3}')
    scene = dataset / "0"
    scene.mkdir()
    (scene / ".zgroup").write_text('{"zarr_format": 2}')
    attrs = {
        "multiscales": [
            {
                "name": "scene",
                "axes": [
                    {"name": "t", "type": "time"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space"},
                    {"name": "x", "type": "space"},
                ],
                "datasets": [
                    {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]}]}
                ],
            }
        ]
    }
    (scene / ".zattrs").write_text(json.dumps(attrs))
    ome = dataset / "OME"
    ome.mkdir()
    (ome / ".zgroup").write_text('{"zarr_format": 2}')
    (ome / ".zattrs").write_text('{"series": ["0"]}')
    (ome / "METADATA.ome.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="2" SizeX="2"/></Image></OME>',
        encoding="utf-8",
    )

    try:
        ds = open_dataset(dataset)
        report = ds.inspect_axis_metadata("0")
        assert any(message.code == "y_placeholder" for message in report.errors)
        assert any(message.code == "x_placeholder" for message in report.errors)
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_build_plane_table_warns_on_plane_mismatch(vsi_subset):
    xml_path = vsi_subset / "OME" / "METADATA.ome.xml"
    text = xml_path.read_text(encoding="utf-8")
    xml_path.write_text(
        text.replace(
            "</Pixels>",
            '<Plane TheT="101" TheC="0" TheZ="0" PositionZ="1.0" PositionZUnit="痠"/></Pixels>',
            1,
        ),
        encoding="utf-8",
    )

    ds = open_dataset(vsi_subset)
    _, report = ds.build_plane_table("0", persist=False)
    assert any(message.code == "plane_count_mismatch" for message in report.warnings)


def test_no_t_scalar_repair_from_lif_delta_t(lif_subset):
    ds = open_dataset(lif_subset, mode="a")
    repair = ds.repair_axis_metadata("15", persist=True)
    assert repair.axis_states["t"].repaired is False
    assert repair.persisted is True


def test_missing_ome_xml_disables_axis_repair_but_allows_window_repair(vsi_subset):
    xml_path = vsi_subset / "OME" / "METADATA.ome.xml"
    xml_path.unlink()

    ds = open_dataset(vsi_subset, mode="a")
    report = ds.inspect_axis_metadata("0")
    repaired = ds.repair_axis_metadata("0", persist=True)

    assert any(message.code == "ome_xml_missing" for message in report.warnings)
    assert repaired.axis_states["z"].repaired is False
    assert repaired.persisted is True
    assert ds.read_scene_metadata("0", corrected=False)["omero"]["channels"][0]["window"]["max"] == 65535.0
