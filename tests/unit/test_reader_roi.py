from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_load_roi_returns_array_and_metadata():
    dataset = _fresh_dataset_path("reader_roi")
    try:
        _create_small_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        ds.write_roi("0", "roi_1", {"t": (0, 2), "z": 0, "y": (2, 6), "x": (3, 7)}, attrs={"note": "test-roi"})

        reopened = open_dataset(dataset)
        result = reopened.load_roi("0", "roi_1")
        metadata = reopened.read_roi_metadata("0", "roi_1")

        assert reopened.list_rois("0") == ["roi_1"]
        assert result.shape == (2, 1, 1, 4, 4)
        assert result.array.shape == (2, 1, 1, 4, 4)
        assert result.roi_attrs == {"note": "test-roi"}
        assert result.microio["origin"] == {"t": 0, "c": 0, "z": 0, "y": 2, "x": 3}
        assert result.microio["source_scene_id"] == "0"
        assert result.ome["multiscales"][0]["datasets"][0]["path"] == "0"
        assert metadata["roi_attrs"] == {"note": "test-roi"}
        assert metadata["microio"]["slices"]["z"]["indexed"] is True
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_load_roi_raises_for_unknown_name():
    dataset = _fresh_dataset_path("reader_roi_missing")
    try:
        _create_small_dataset(dataset)
        ds = open_dataset(dataset)
        try:
            ds.load_roi("0", "missing")
        except KeyError as exc:
            assert "available=[]" in str(exc)
        else:
            raise AssertionError("Expected KeyError for missing ROI")
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def _create_small_dataset(path: Path) -> Path:
    root = zarr.open(path, mode="w", zarr_format=3)
    scene = root.create_group("0")
    scene.attrs["ome"] = {
        "version": "0.5",
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
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5, 0.5]}],
                    }
                ],
            }
        ],
    }
    level0 = np.arange(2 * 1 * 2 * 8 * 8, dtype=np.uint16).reshape(2, 1, 2, 8, 8)
    scene.create_array("0", data=level0, dimension_names=("t", "c", "z", "y", "x"))
    ome = root.create_group("OME")
    ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
    (path / "OME" / "METADATA.ome.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="2" SizeC="1" SizeZ="2" SizeY="8" SizeX="8"/></Image></OME>',
        encoding="utf-8",
    )
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
