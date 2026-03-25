from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_complete_plane_delta_t_wins_over_vsi_numbered_timing():
    dataset = _fresh_dataset_path("reader_time_delta_wins")
    try:
        _create_time_dataset(
            dataset,
            plane_deltas=[0.0, 0.0, 10.0, 10.0, 20.0, 20.0],
            vsi_values=[100.0, 100.0, 110.0, 110.0, 120.0, 120.0],
        )
        ds = open_dataset(dataset, mode="a")

        table, report = ds.build_plane_table("0", persist=True, filetype="vsi")
        repair = ds.repair_axis_metadata("0", persist=True, filetype="vsi")
        metadata = ds.read_table_metadata("0")

        assert report.persisted is True
        assert table["positioners_t"].tolist() == [0.0, 0.0, 10.0, 10.0, 20.0, 20.0]
        assert metadata["axis_metadata"]["t"]["source"] == "Plane.DeltaT"
        assert repair.axis_states["t"].source == "Plane.DeltaT"
        assert repair.axis_states["t"].value == 10.0
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_incomplete_plane_delta_t_switches_entirely_to_vsi_without_mixing():
    dataset = _fresh_dataset_path("reader_time_vsi_fallback")
    try:
        _create_time_dataset(
            dataset,
            plane_deltas=[0.0, None, 10.0, 10.0, 20.0, 20.0],
            vsi_values=[100.0, 100.0, 110.0, 110.0, 120.0, 120.0],
        )
        ds = open_dataset(dataset, mode="a")

        table, _ = ds.build_plane_table("0", persist=True, filetype="vsi")
        repair = ds.repair_axis_metadata("0", persist=True, filetype="vsi")
        metadata = ds.read_table_metadata("0")

        assert table["positioners_t"].tolist() == [100.0, 100.0, 110.0, 110.0, 120.0, 120.0]
        assert metadata["axis_metadata"]["t"]["source"] == "OriginalMetadata.VSI"
        assert repair.axis_states["t"].source == "OriginalMetadata.VSI"
        assert repair.axis_states["t"].value == 10.0
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_vsi_timing_is_ignored_without_vsi_filetype():
    dataset = _fresh_dataset_path("reader_time_generic")
    try:
        _create_time_dataset(dataset, plane_deltas=None, vsi_values=[0.0, 0.0, 10.0, 10.0, 20.0, 20.0])
        ds = open_dataset(dataset, mode="a")

        table, _ = ds.build_plane_table("0", persist=False)
        repair = ds.repair_axis_metadata("0", persist=False)

        assert np.isnan(table["positioners_t"]).all()
        assert repair.axis_states["t"].repaired is False
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_incomplete_vsi_timing_is_rejected_without_mixing():
    dataset = _fresh_dataset_path("reader_time_vsi_incomplete")
    try:
        _create_time_dataset(dataset, plane_deltas=None, vsi_values=[0.0, 0.0, 10.0, 10.0, 20.0])
        ds = open_dataset(dataset, mode="a")

        table, report = ds.build_plane_table("0", persist=False, filetype="vsi")
        repair = ds.repair_axis_metadata("0", persist=False, filetype="vsi")

        assert np.isnan(table["positioners_t"]).all()
        assert any(message.code == "vsi_time_incomplete" for message in report.warnings)
        assert repair.axis_states["t"].repaired is False
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_inconsistent_plane_timing_skips_scalar_t_repair_but_keeps_table_values():
    dataset = _fresh_dataset_path("reader_time_inconsistent")
    try:
        _create_time_dataset(dataset, plane_deltas=None, vsi_values=[0.0, 0.0, 10.0, 10.0, 25.0, 25.0])
        ds = open_dataset(dataset, mode="a")

        table, _ = ds.build_plane_table("0", persist=False, filetype="vsi")
        repair = ds.repair_axis_metadata("0", persist=False, filetype="vsi")

        assert table["positioners_t"].tolist() == [0.0, 0.0, 10.0, 10.0, 25.0, 25.0]
        assert repair.axis_states["t"].repaired is False
        assert any(message.code == "t_inconsistent_plane_spacing" for message in repair.warnings)
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_pixels_time_increment_remains_preferred_scalar_source():
    dataset = _fresh_dataset_path("reader_time_increment")
    try:
        _create_time_dataset(dataset, plane_deltas=None, vsi_values=[0.0, 0.0, 10.0, 10.0, 20.0, 20.0], time_increment=7.5)
        ds = open_dataset(dataset, mode="a")

        repair = ds.repair_axis_metadata("0", persist=False, filetype="vsi")

        assert repair.axis_states["t"].repaired is True
        assert repair.axis_states["t"].source == "Pixels.TimeIncrement"
        assert repair.axis_states["t"].value == 7.5
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def _create_time_dataset(
    path: Path,
    *,
    size_t: int = 3,
    size_c: int = 1,
    size_z: int = 2,
    plane_deltas: list[float | None] | None,
    vsi_values: list[float] | None,
    vsi_unit: str = "10^-3s^1",
    time_increment: float | None = None,
) -> Path:
    root = zarr.open(path, mode="w")
    scene = root.create_group("0")
    scene.attrs.update(
        {
            "multiscales": [
                {
                    "name": "scene",
                    "axes": [
                        {"name": "t", "type": "time"},
                        {"name": "c", "type": "channel"},
                        {"name": "z", "type": "space", "unit": "micrometer"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]}],
                        }
                    ],
                }
            ]
        }
    )
    scene.create_array("0", data=np.zeros((size_t, size_c, size_z, 4, 4), dtype=np.uint16))
    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(
        _ome_xml(
            size_t=size_t,
            size_c=size_c,
            size_z=size_z,
            plane_deltas=plane_deltas,
            vsi_values=vsi_values,
            vsi_unit=vsi_unit,
            time_increment=time_increment,
        ),
        encoding="utf-8",
    )
    return path


def _ome_xml(
    *,
    size_t: int,
    size_c: int,
    size_z: int,
    plane_deltas: list[float | None] | None,
    vsi_values: list[float] | None,
    vsi_unit: str,
    time_increment: float | None,
) -> str:
    expected = size_t * size_c * size_z
    plane_deltas = plane_deltas or [None] * expected
    planes: list[str] = []
    flat_index = 0
    for t_index in range(size_t):
        for c_index in range(size_c):
            for z_index in range(size_z):
                delta = plane_deltas[flat_index] if flat_index < len(plane_deltas) else None
                delta_attr = ""
                if delta is not None:
                    delta_attr = f' DeltaT="{delta}" DeltaTUnit="{vsi_unit}"'
                planes.append(
                    f'<Plane TheT="{t_index}" TheC="{c_index}" TheZ="{z_index}" PositionZ="{z_index}" '
                    f'PositionZUnit="µm"{delta_attr}/>'
                )
                flat_index += 1

    annotations: list[str] = []
    if vsi_values is not None:
        for index, value in enumerate(vsi_values, start=1):
            annotations.append(
                f'<XMLAnnotation ID="Annotation:V{index}" Namespace="openmicroscopy.org/OriginalMetadata">'
                f"<Value><OriginalMetadata><Key>scene Value #{index:04d}</Key><Value>{value}</Value></OriginalMetadata></Value>"
                f"</XMLAnnotation>"
            )
            annotations.append(
                f'<XMLAnnotation ID="Annotation:U{index}" Namespace="openmicroscopy.org/OriginalMetadata">'
                f"<Value><OriginalMetadata><Key>scene Units #{index:04d}</Key><Value>{vsi_unit}</Value></OriginalMetadata></Value>"
                f"</XMLAnnotation>"
            )

    time_increment_attr = ""
    if time_increment is not None:
        time_increment_attr = f' TimeIncrement="{time_increment}" TimeIncrementUnit="s"'

    return (
        '<?xml version="1.0" encoding="UTF-8"?>'
        '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
        '<Image Name="scene">'
        f'<Pixels SizeT="{size_t}" SizeC="{size_c}" SizeZ="{size_z}" SizeY="4" SizeX="4"'
        f' PhysicalSizeZ="2.0" PhysicalSizeZUnit="µm"{time_increment_attr}>'
        f'{"".join(planes)}'
        "</Pixels>"
        "</Image>"
        f'<StructuredAnnotations>{"".join(annotations)}</StructuredAnnotations>'
        "</OME>"
    )


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
