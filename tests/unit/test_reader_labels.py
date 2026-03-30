from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import dask.array as da
import numpy as np
import pytest
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_list_labels_and_accessor_on_written_label_dataset():
    dataset = _fresh_dataset_path("reader_labels")
    try:
        _create_small_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        ds.write_label_image(
            "0",
            "segmentation",
            np.ones((2, 1, 2, 8, 8), dtype=np.uint16),
            attrs={"kind": "mask"},
            colors=[{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 1, "rgba": [0, 255, 0, 255]}],
            properties=[{"label-value": 1, "class": "mask"}],
        )

        reopened = open_dataset(dataset)
        assert reopened.list_labels("0") == ["segmentation"]

        label = reopened.get_label("0", "segmentation")
        metadata = label.metadata()
        levels = label.levels()

        assert label.scene_ref.id == "0"
        assert metadata.label_name == "segmentation"
        assert metadata.microio["label-attrs"] == {"kind": "mask"}
        assert metadata.label_attrs == {"kind": "mask"}
        assert metadata.colors == [{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 1, "rgba": [0, 255, 0, 255]}]
        assert metadata.properties == [{"label-value": 1, "class": "mask"}]
        assert metadata.ome["image-label"]["source"]["image"] == "../../"
        assert levels[0].container_kind == "label"
        assert levels[0].container_name == "segmentation"
        assert levels[0].path == "0"
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_label_level_access_by_index_and_path():
    dataset = _fresh_dataset_path("reader_labels_levels")
    try:
        _create_small_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        label = np.zeros((2, 1, 2, 8, 8), dtype=np.uint16)
        label[..., 2:6, 2:6] = 4
        ds.write_label_image("0", "segmentation", label)

        reopened = open_dataset(dataset)
        levels = reopened.list_label_levels("0", "segmentation")
        by_index = reopened.label_level_ref("0", "segmentation", 1)
        by_path = reopened.label_level_ref("0", "segmentation", "1")
        arr = reopened.read_label("0", "segmentation", 0)
        raw = reopened.read_label_zarr("0", "segmentation", 0)
        eager = reopened.read_label_numpy("0", "segmentation", 0)

        assert by_index == by_path
        assert len(levels) == 2
        assert isinstance(arr, da.Array)
        assert isinstance(raw, zarr.Array)
        assert isinstance(eager, np.ndarray)
        assert arr.shape == levels[0].shape
        assert raw.shape == levels[0].shape
        assert eager.shape == levels[0].shape
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_missing_labels_are_reported_cleanly(vsi_subset):
    ds = open_dataset(vsi_subset)

    assert ds.list_labels("0") == []
    with pytest.raises(KeyError, match="available=\\[\\]"):
        ds.get_label("0", "missing").metadata()


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
                    {"path": "0", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 0.5, 0.5]}]},
                    {"path": "1", "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}]},
                ],
            }
        ],
    }
    level0 = np.zeros((2, 1, 2, 8, 8), dtype=np.uint16)
    scene.create_array("0", data=level0, dimension_names=("t", "c", "z", "y", "x"))
    scene.create_array("1", data=level0[..., ::2, ::2], dimension_names=("t", "c", "z", "y", "x"))
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
