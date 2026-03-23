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


def test_write_label_image_rejects_non_integer_dtype():
    dataset = _fresh_dataset_path("writer_label_dtype")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        with pytest.raises(ValueError, match="integer dtype"):
            ds.write_label_image("0", "mask", np.zeros((1, 1, 1, 4, 4), dtype=np.float32))
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_image_nests_user_attrs_under_microio():
    dataset = _fresh_dataset_path("writer_label_attrs")
    try:
        _create_writable_dataset(dataset)
        ds = open_dataset(dataset, mode="a")
        ds.write_label_image("0", "mask", np.zeros((1, 1, 1, 4, 4), dtype=np.uint16), attrs={"kind": "segmentation"})
        label_group = ds.root["0"]["labels"]["mask"]
        assert label_group.attrs["microio"]["label-attrs"] == {"kind": "segmentation"}
        assert "kind" not in label_group.attrs
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


def test_write_label_image_accepts_singleton_channel_for_multichannel_source():
    dataset = _fresh_dataset_path("writer_label_singleton_channel")
    try:
        _create_writable_dataset(dataset, channels=2)
        ds = open_dataset(dataset, mode="a")
        label_data = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
        label_data[..., 1:3, 1:3] = 2
        report = ds.write_label_image("0", "mask", label_data)
        label_group = ds.root["0"]["labels"]["mask"]
        assert report.shape == (1, 1, 1, 4, 4)
        assert label_group["0"].shape == (1, 1, 1, 4, 4)
        assert label_group.attrs["microio"]["channel_mode"] == "singleton"
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_image_rejects_invalid_channel_size():
    dataset = _fresh_dataset_path("writer_label_bad_channel")
    try:
        _create_writable_dataset(dataset, channels=2)
        ds = open_dataset(dataset, mode="a")
        with pytest.raises(ValueError, match="channel axis"):
            ds.write_label_image("0", "mask", np.zeros((1, 3, 1, 4, 4), dtype=np.uint16))
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


def test_write_label_timepoint_initializes_and_tracks_written_frames():
    dataset = _fresh_dataset_path("writer_label_timepoint")
    try:
        _create_writable_dataset(dataset, levels=2, times=3)
        ds = open_dataset(dataset, mode="a")
        first = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
        first[..., 0:2, 0:2] = 3
        report = ds.write_label_timepoint("0", "mask", first, timepoint=1)

        label_group = ds.root["0"]["labels"]["mask"]
        assert report.written_timepoint == 1
        assert report.initialized is True
        assert label_group["0"].shape == (3, 1, 1, 4, 4)
        assert label_group["0"][1, 0, 0, 0:2, 0:2].tolist() == [[3, 3], [3, 3]]
        assert int(label_group["0"][0, 0, 0, 0, 0]) == 0
        assert label_group.attrs["microio"]["written_timepoints"] == [1]
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_timepoint_rejects_duplicate_without_overwrite():
    dataset = _fresh_dataset_path("writer_label_timepoint_duplicate")
    try:
        _create_writable_dataset(dataset, times=3)
        ds = open_dataset(dataset, mode="a")
        frame = np.ones((1, 1, 1, 4, 4), dtype=np.uint16)
        ds.write_label_timepoint("0", "mask", frame, timepoint=0)
        with pytest.raises(FileExistsError, match="overwrite_timepoint=True"):
            ds.write_label_timepoint("0", "mask", frame, timepoint=0)
        ds.write_label_timepoint("0", "mask", np.full((1, 1, 1, 4, 4), 2, dtype=np.uint16), timepoint=0, overwrite_timepoint=True)
        assert int(ds.root["0"]["labels"]["mask"]["0"][0, 0, 0, 0, 0]) == 2
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_timepoint_rejects_metadata_updates_on_existing_label():
    dataset = _fresh_dataset_path("writer_label_timepoint_metadata")
    try:
        _create_writable_dataset(dataset, times=2)
        ds = open_dataset(dataset, mode="a")
        frame = np.ones((1, 1, 1, 4, 4), dtype=np.uint16)
        ds.write_label_timepoint("0", "mask", frame, timepoint=0)
        with pytest.raises(ValueError, match="cannot accept dtype, chunks, attrs, colors, or properties"):
            ds.write_label_timepoint("0", "mask", frame, timepoint=1, attrs={"kind": "second"})
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_image_defaults_to_source_chunks():
    dataset = _fresh_dataset_path("writer_label_source_chunks")
    try:
        _create_writable_dataset(dataset, zarr_format=3, times=3, channels=2)
        ds = open_dataset(dataset, mode="a")
        label_data = np.zeros((3, 1, 1, 4, 4), dtype=np.uint16)
        ds.write_label_image("0", "mask", label_data)

        chunks = tuple(int(chunk) for chunk in ds.root["0"]["labels"]["mask"]["0"].chunks)
        assert chunks == (3, 1, 1, 4, 4)
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_timepoint_initializes_array_with_single_timepoint_chunks():
    dataset = _fresh_dataset_path("writer_label_timepoint_chunks")
    try:
        root = zarr.open(dataset, mode="w", zarr_format=3)
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
        scene.create_array(
            "0",
            shape=(100, 1, 34, 2800, 2800),
            dtype=np.uint16,
            chunks=(1, 1, 1, 256, 256),
            dimension_names=("t", "c", "z", "y", "x"),
            write_data=False,
        )
        ome = root.create_group("OME")
        ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
        (dataset / "OME" / "METADATA.ome.xml").write_text(
            '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="100" SizeC="1" SizeZ="34" SizeY="2800" SizeX="2800"/></Image></OME>',
            encoding="utf-8",
        )

        ds = open_dataset(dataset, mode="a")
        frame = da.zeros((1, 1, 34, 2800, 2800), chunks=(1, 1, 1, 700, 700), dtype=np.uint16)
        ds.write_label_timepoint("0", "mask", frame, timepoint=0)

        chunks = tuple(int(chunk) for chunk in ds.root["0"]["labels"]["mask"]["0"].chunks)
        assert chunks == (1, 1, 1, 256, 256)
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def test_write_label_image_creates_label_listing_and_multiscale_levels():
    dataset = _fresh_dataset_path("writer_label_multiscale")
    try:
        _create_writable_dataset(dataset, zarr_format=3, levels=2)
        ds = open_dataset(dataset, mode="a")
        label_data = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
        label_data[..., 1:3, 1:3] = 4

        report = ds.write_label_image(
            "0",
            "mask",
            label_data,
            colors=[{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 4, "rgba": [255, 0, 0, 255]}],
            properties=[{"label-value": 4, "class": "nucleus"}],
        )

        scene_labels = ds.root["0"]["labels"]
        label_group = scene_labels["mask"]
        assert report.level_path == "0"
        assert scene_labels.attrs["ome"]["labels"] == ["mask"]
        assert label_group["0"].shape == (1, 1, 1, 4, 4)
        assert label_group["1"].shape == (1, 1, 1, 2, 2)
        assert label_group.attrs["ome"]["image-label"]["source"]["image"] == "../../"
        assert label_group.attrs["ome"]["image-label"]["colors"][1]["label-value"] == 4
        assert len(label_group.attrs["ome"]["multiscales"][0]["datasets"]) == 2
        assert tuple(label_group["0"].metadata.dimension_names) == ("t", "c", "z", "y", "x")
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


def test_write_roi_does_not_emit_label_metadata():
    dataset = _fresh_dataset_path("writer_roi_metadata")
    try:
        _create_writable_dataset(dataset, zarr_format=3)
        ds = open_dataset(dataset, mode="a")
        ds.write_roi("0", "roi", {"y": (0, 2), "x": (0, 2)})
        roi_group = ds.root["0"]["rois"]["roi"]
        assert "image-label" not in roi_group.attrs.get("ome", {})
        assert len(roi_group.attrs["ome"]["multiscales"][0]["datasets"]) == 1
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


def test_write_label_image_overwrite_refreshes_listing():
    dataset = _fresh_dataset_path("writer_label_overwrite")
    try:
        _create_writable_dataset(dataset, zarr_format=2)
        ds = open_dataset(dataset, mode="a")
        first = np.zeros((1, 1, 1, 4, 4), dtype=np.uint16)
        second = np.ones((1, 1, 1, 4, 4), dtype=np.uint16)
        ds.write_label_image("0", "mask", first)
        ds.write_label_image("0", "mask", second, overwrite=True)
        labels = ds.root["0"]["labels"]
        assert labels.attrs["labels"] == ["mask"]
        assert ds.root["0"]["labels"]["mask"]["0"][0, 0, 0, 0, 0] == 1
    finally:
        shutil.rmtree(dataset, ignore_errors=True)


def _create_writable_dataset(path: Path, *, zarr_format: int = 3, levels: int = 1, times: int = 1, channels: int = 1) -> Path:
    root = zarr.open(path, mode="w", zarr_format=zarr_format)
    scene = root.create_group("0")
    multiscales = [
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
                    "path": str(level),
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 2.0, 0.5 * (2**level), 0.5 * (2**level)]}],
                }
                for level in range(levels)
            ],
            "version": "0.4",
        }
    ]
    if zarr_format >= 3:
        scene.attrs["ome"] = {"version": "0.5", "multiscales": multiscales}
    else:
        scene.attrs["multiscales"] = multiscales

    level0 = np.arange(times * channels * 1 * 4 * 4, dtype=np.uint16).reshape(times, channels, 1, 4, 4)
    scene.create_array(
        "0",
        data=level0,
        dimension_names=("t", "c", "z", "y", "x") if zarr_format >= 3 else None,
    )
    if levels > 1:
        scene.create_array(
            "1",
            data=level0[..., ::2, ::2],
            dimension_names=("t", "c", "z", "y", "x") if zarr_format >= 3 else None,
        )
    ome = root.create_group("OME")
    if zarr_format >= 3:
        ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
    else:
        ome.attrs["series"] = ["0"]
    (path / "OME" / "METADATA.ome.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image Name="scene"><Pixels SizeT="{times}" SizeC="{channels}" SizeZ="1" SizeY="4" SizeX="4"/></Image></OME>',
        encoding="utf-8",
    )
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
