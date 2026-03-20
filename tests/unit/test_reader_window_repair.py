from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_repair_updates_stale_windows_with_dtype_bounds():
    dataset = _fresh_dataset_path("window_repair_stale")
    try:
        data = np.arange(2 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 1, 2, 4, 4)
        _create_dataset(dataset, data=data, omero={"channels": [{"color": "FFFFFF", "window": {"min": 0, "max": 10, "start": 1, "end": 9}}]})

        ds = open_dataset(dataset, mode="a")
        report = ds.repair_axis_metadata("0", persist=True)
        repaired = ds.read_scene_metadata("0", corrected=False)["omero"]["channels"][0]["window"]

        assert report.persisted is True
        assert repaired == {"min": 0.0, "max": 65535.0, "start": 0.0, "end": float(data.max())}
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_creates_minimal_omero_and_defaults():
    dataset = _fresh_dataset_path("window_repair_missing_omero")
    try:
        data = np.stack(
            [
                np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4),
                np.arange(2 * 2 * 4 * 4, dtype=np.uint16).reshape(2, 2, 4, 4) + 100,
            ],
            axis=1,
        )
        _create_dataset(dataset, data=data, omero=None)

        ds = open_dataset(dataset, mode="a")
        ds.repair_axis_metadata("0", persist=True)
        omero = ds.read_scene_metadata("0", corrected=False)["omero"]

        assert len(omero["channels"]) == 2
        assert omero["rdefs"] == {"defaultT": 0, "defaultZ": 0, "model": "color"}
        assert omero["channels"][0]["color"] == "00FF00"
        assert omero["channels"][1]["color"] == "FF0000"
        assert omero["channels"][0]["window"]["max"] == 65535.0
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_rebuilds_channel_count_mismatch():
    dataset = _fresh_dataset_path("window_repair_channel_mismatch")
    try:
        data = np.arange(1 * 2 * 1 * 4 * 4, dtype=np.uint16).reshape(1, 2, 1, 4, 4)
        _create_dataset(dataset, data=data, omero={"channels": [{"color": "FFFFFF", "window": {"min": 0, "max": 1, "start": 0, "end": 1}}]})

        ds = open_dataset(dataset, mode="a")
        report = ds.repair_axis_metadata("0", persist=True)
        omero = ds.read_scene_metadata("0", corrected=False)["omero"]

        assert any(message.code == "omero_channels_rebuilt" for message in report.warnings)
        assert len(omero["channels"]) == 2
        assert all(channel["window"]["max"] == 65535.0 for channel in omero["channels"])
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_uses_float_dtype_bounds():
    dataset = _fresh_dataset_path("window_repair_float32")
    try:
        data = np.linspace(0.0, 1.0, num=32, dtype=np.float32).reshape(1, 1, 2, 4, 4)
        _create_dataset(dataset, data=data, omero=None)

        ds = open_dataset(dataset, mode="a")
        ds.repair_axis_metadata("0", persist=True)
        window = ds.read_scene_metadata("0", corrected=False)["omero"]["channels"][0]["window"]
        finfo = np.finfo(np.float32)

        assert window["start"] == 0.0
        assert window["end"] == 1.0
        assert window["min"] == float(finfo.min)
        assert window["max"] == float(finfo.max)
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_skips_unsupported_dtype():
    dataset = _fresh_dataset_path("window_repair_bool")
    try:
        data = np.zeros((1, 1, 1, 4, 4), dtype=bool)
        _create_dataset(dataset, data=data, omero=None)

        ds = open_dataset(dataset, mode="a")
        report = ds.repair_axis_metadata("0", persist=True)

        assert report.persisted is False
        assert any(message.code == "omero_window_dtype_unsupported" for message in report.warnings)
        assert ds.read_scene_metadata("0", corrected=False).get("omero") is None
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_does_not_persist_when_requested_false():
    dataset = _fresh_dataset_path("window_repair_no_persist")
    try:
        data = np.arange(1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
        _create_dataset(dataset, data=data, omero={"channels": [{"color": "FFFFFF", "window": {"min": 0, "max": 1, "start": 0, "end": 1}}]})

        ds = open_dataset(dataset, mode="a")
        report = ds.repair_axis_metadata("0", persist=False)
        window = ds.read_scene_metadata("0", corrected=False)["omero"]["channels"][0]["window"]

        assert report.persisted is False
        assert window == {"min": 0, "max": 1, "start": 0, "end": 1}
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_repair_windows_persist_without_ome_xml():
    dataset = _fresh_dataset_path("window_repair_no_xml")
    try:
        data = np.arange(1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
        _create_dataset(dataset, data=data, omero={"channels": [{"color": "FFFFFF", "window": {"min": 0, "max": 1, "start": 0, "end": 1}}]})
        (dataset / "OME" / "METADATA.ome.xml").unlink()

        ds = open_dataset(dataset, mode="a")
        report = ds.repair_axis_metadata("0", persist=True)
        window = ds.read_scene_metadata("0", corrected=False)["omero"]["channels"][0]["window"]

        assert report.persisted is True
        assert any(message.code == "ome_xml_missing" for message in report.warnings)
        assert window["max"] == 65535.0
        assert window["end"] == float(data.max())
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_v3_scene_repair_preserves_ome_namespace():
    dataset = _fresh_dataset_path("window_repair_v3")
    try:
        data = np.arange(1 * 1 * 1 * 4 * 4, dtype=np.uint16).reshape(1, 1, 1, 4, 4)
        _create_dataset(dataset, data=data, omero=None, zarr_format=3)

        ds = open_dataset(dataset, mode="a")
        ds.repair_axis_metadata("0", persist=True)
        raw = ds.root["0"].attrs.asdict()

        assert "ome" in raw
        assert "multiscales" not in raw
        assert raw["ome"]["omero"]["channels"][0]["window"]["max"] == 65535.0
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def _create_dataset(path: Path, *, data: np.ndarray, omero: dict | None, zarr_format: int = 2) -> None:
    root = zarr.open(path, mode="w", zarr_format=zarr_format)
    scene = root.create_group("0")
    scene_attrs = {
        "multiscales": [
            {
                "name": "scene",
                "axes": [
                    {"name": "t", "type": "time", "unit": "second"},
                    {"name": "c", "type": "channel"},
                    {"name": "z", "type": "space", "unit": "micrometer"},
                    {"name": "y", "type": "space", "unit": "micrometer"},
                    {"name": "x", "type": "space", "unit": "micrometer"},
                ],
                "datasets": [
                    {
                        "path": "0",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                    }
                ],
                "version": "0.4" if zarr_format == 2 else "0.5",
            }
        ]
    }
    if omero is not None:
        scene_attrs["omero"] = omero
    if zarr_format >= 3:
        scene.attrs["ome"] = {"version": "0.5", **scene_attrs}
    else:
        scene.attrs.update(scene_attrs)
    scene.create_array(
        "0",
        data=data,
        chunks=tuple(min(dim, chunk) for dim, chunk in zip(data.shape, (1, 1, 1, 2, 2), strict=True)),
        dimension_names=("t", "c", "z", "y", "x") if zarr_format >= 3 else None,
    )
    ome = root.create_group("OME")
    if zarr_format >= 3:
        ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
    else:
        ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            f'<Image Name="scene"><Pixels SizeT="{data.shape[0]}" SizeC="{data.shape[1]}" '
            f'SizeZ="{data.shape[2]}" SizeY="{data.shape[3]}" SizeX="{data.shape[4]}" '
            'PhysicalSizeZ="1.0" PhysicalSizeZUnit="micrometer" TimeIncrement="1.0" TimeIncrementUnit="second"/></Image>'
            "</OME>"
        ),
        encoding="utf-8",
    )


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
