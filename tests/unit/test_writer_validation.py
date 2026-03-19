from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_write_table_from_column_mapping():
    dataset = _fresh_dataset_path("writer_table_map")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        report = ds.write_table("0", "measurements", {"a": [1, 2], "b": [3.0, 4.0]}, attrs={"kind": "test"})
        table = ds.root["0"]["tables"]["measurements"]

        assert report.persisted is True
        assert report.row_count == 2
        assert table.attrs["schema"] == "microio.table"
        assert table.attrs["n_rows"] == 2
        assert table.attrs["kind"] == "test"
        assert table["a"][:].tolist() == [1, 2]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_table_from_row_records():
    dataset = _fresh_dataset_path("writer_table_rows")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        report = ds.write_table("0", "records", [{"x": 1, "y": 2}, {"x": 3, "y": 4}])
        assert report.column_names == ["x", "y"]
        assert ds.root["0"]["tables"]["records"]["y"][:].tolist() == [2, 4]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_table_rejects_nested_object_column():
    dataset = _fresh_dataset_path("writer_table_nested")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        with pytest.raises(ValueError, match="nested objects"):
            ds.write_table("0", "bad", {"a": [{"x": 1}]})
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_table_append_requires_matching_schema():
    dataset = _fresh_dataset_path("writer_table_append")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        ds.write_table("0", "measurements", {"a": [1], "b": [2]})
        ds.write_table("0", "measurements", {"a": [3], "b": [4]}, append=True)
        assert ds.root["0"]["tables"]["measurements"]["a"][:].tolist() == [1, 3]
        with pytest.raises(ValueError, match="cannot append"):
            ds.write_table("0", "measurements", {"a": [5]}, append=True)
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_table_supports_pandas_if_installed():
    pd = pytest.importorskip("pandas")
    dataset = _fresh_dataset_path("writer_table_pandas")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        frame = pd.DataFrame({"t": [0, 1], "label": [11, 12]})
        report = ds.write_table("0", "frame", frame)
        assert report.row_count == 2
        assert ds.root["0"]["tables"]["frame"]["label"][:].tolist() == [11, 12]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_image_rejects_shape_mismatch():
    dataset = _fresh_dataset_path("writer_label_shape")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        with pytest.raises(ValueError, match="does not match source level shape"):
            ds.write_label_image("0", "mask", np.zeros((1, 1, 2, 4, 4), dtype=np.uint16))
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_methods_reject_read_only_handle():
    dataset = _fresh_dataset_path("writer_readonly")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset)
        with pytest.raises(PermissionError):
            ds.write_table("0", "measurements", {"a": [1]})
        with pytest.raises(PermissionError):
            ds.write_label_image("0", "mask", np.zeros((1, 1, 1, 4, 4), dtype=np.uint16))
        with pytest.raises(PermissionError):
            ds.write_roi("0", "roi", {"y": (0, 2), "x": (0, 2)})
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_roi_normalizes_integer_axis_without_squeezing():
    dataset = _fresh_dataset_path("writer_roi_index")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        report = ds.write_roi("0", "roi", {"t": 0, "z": 0, "y": (1, 3), "x": (0, 2)})
        roi = ds.root["0"]["rois"]["roi"]["0"]
        assert report.shape == (1, 1, 1, 2, 2)
        assert roi.shape == (1, 1, 1, 2, 2)
        assert ds.root["0"]["rois"]["roi"].attrs["microio"]["slices"]["t"]["indexed"] is True
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_roi_rejects_out_of_bounds_slice():
    dataset = _fresh_dataset_path("writer_roi_bounds")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        with pytest.raises(ValueError, match="out of bounds"):
            ds.write_roi("0", "roi", {"y": (0, 9)})
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_overwrite_guards_apply_to_tables_labels_and_rois():
    dataset = _fresh_dataset_path("writer_overwrite")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        ds.write_table("0", "measurements", {"a": [1]})
        ds.write_label_image("0", "mask", np.zeros((1, 1, 1, 4, 4), dtype=np.uint16))
        ds.write_roi("0", "roi", {"y": (0, 2), "x": (0, 2)})
        with pytest.raises(FileExistsError):
            ds.write_table("0", "measurements", {"a": [2]})
        with pytest.raises(FileExistsError):
            ds.write_label_image("0", "mask", np.zeros((1, 1, 1, 4, 4), dtype=np.uint16))
        with pytest.raises(FileExistsError):
            ds.write_roi("0", "roi", {"y": (0, 2), "x": (0, 2)})
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def _create_writable_dataset(path: Path) -> Path:
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
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5, 0.5]}],
                        }
                    ],
                    "version": "0.4",
                }
            ]
        }
    )
    scene.create_array("0", data=np.arange(16, dtype=np.uint16).reshape(1, 1, 1, 4, 4))
    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="4" SizeX="4"/></Image></OME>',
        encoding="utf-8",
    )
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
