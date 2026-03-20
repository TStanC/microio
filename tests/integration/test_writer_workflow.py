from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_write_table_label_and_roi_on_small_dataset():
    dataset = _fresh_dataset_path("writer_integration")
    try:
        _create_small_dataset(dataset)
        ds = open_dataset(dataset, mode="a")

        table_report = ds.write_table(
            "0",
            "nuclei",
            {"timepoint": [0, 0], "label_id": [10, 11], "volume": [12.5, 14.0]},
            attrs={"description": "derived measurements"},
        )
        label_data = np.zeros(ds.level_ref("0", 0).shape, dtype=np.uint16)
        label_data[:, :, :, 1:3, 2:5] = 7
        label_report = ds.write_label_image(
            "0",
            "segmentation",
            label_data,
            colors=[{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 7, "rgba": [255, 255, 0, 255]}],
        )
        roi_report = ds.write_roi("0", "roi_1", {"t": (0, 2), "z": (0, 2), "y": (2, 8), "x": (3, 9)})

        reopened = open_dataset(dataset)
        table = reopened.root["0"]["tables"]["nuclei"]
        label_group = reopened.root["0"]["labels"]["segmentation"]
        roi_group = reopened.root["0"]["rois"]["roi_1"]

        assert table_report.row_count == 2
        assert table.attrs["description"] == "derived measurements"
        assert table["label_id"][:].tolist() == [10, 11]

        assert label_report.shape == label_data.shape
        assert label_group["0"].shape == label_data.shape
        assert label_group["1"].shape == (2, 1, 3, 6, 6)
        assert label_group.attrs["microio"]["source_scene_id"] == "0"
        assert label_group.attrs["ome"]["image-label"]["source"]["image"] == "../../"
        assert tuple(label_group.attrs["ome"]["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"]) == reopened.level_ref("0", 0).scale
        assert len(label_group.attrs["ome"]["multiscales"][0]["datasets"]) == 2

        assert roi_report.shape == (2, 1, 2, 6, 6)
        assert roi_group["0"].shape == (2, 1, 2, 6, 6)
        assert roi_group.attrs["microio"]["origin"] == {"t": 0, "c": 0, "z": 0, "y": 2, "x": 3}
        assert roi_group.attrs["microio"]["source_scene_id"] == "0"
        assert "image-label" not in roi_group.attrs["ome"]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_table_append_on_vsi_subset(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")
    ds.write_table("0", "measurements", {"a": [1], "b": [2]})
    report = ds.write_table("0", "measurements", {"a": [3, 4], "b": [5, 6]}, append=True)

    reopened = open_dataset(vsi_subset)
    table = reopened.root["0"]["tables"]["measurements"]
    assert report.appended is True
    assert table.attrs["n_rows"] == 3
    assert table["a"][:].tolist() == [1, 3, 4]


def test_write_label_and_roi_on_v2_fixture(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")
    shape = ds.level_ref("0", 0).shape
    label_data = np.zeros(shape, dtype=np.uint16)
    label_data[..., 2:4, 2:4] = 3

    ds.write_label_image("0", "segmentation", label_data)
    ds.write_roi("0", "roi_box", {"t": 0, "z": (0, 2), "y": (0, 8), "x": (0, 8)})

    reopened = open_dataset(vsi_subset)
    labels = reopened.root["0"]["labels"]
    label_group = labels["segmentation"]
    roi_group = reopened.root["0"]["rois"]["roi_box"]

    assert labels.attrs["labels"] == ["segmentation"]
    assert "ome" not in labels.attrs
    assert "image-label" in label_group.attrs
    assert len(label_group.attrs["multiscales"][0]["datasets"]) == len(reopened.list_levels("0"))
    assert "image-label" not in roi_group.attrs


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
                    },
                    {
                        "path": "1",
                        "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 1.0, 1.0]}],
                    },
                ],
            }
        ],
    }
    level0 = np.arange(2 * 1 * 3 * 12 * 12, dtype=np.uint16).reshape(2, 1, 3, 12, 12)
    scene.create_array("0", data=level0, dimension_names=("t", "c", "z", "y", "x"))
    scene.create_array("1", data=level0[..., ::2, ::2], dimension_names=("t", "c", "z", "y", "x"))
    ome = root.create_group("OME")
    ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
    (path / "OME" / "METADATA.ome.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="2" SizeC="1" SizeZ="3" SizeY="12" SizeX="12"/></Image></OME>',
        encoding="utf-8",
    )
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
