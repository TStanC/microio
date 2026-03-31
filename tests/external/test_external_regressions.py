from __future__ import annotations

import json

import dask.array as da
import numpy as np
import pytest

from microio.reader.open import open_dataset


pytestmark = pytest.mark.external_dataset


def test_external_lif_dataset_preserves_duplicate_name_and_invalid_scene_behavior(external_lif_path):
    ds = open_dataset(external_lif_path)

    matches = ds.scene_name_matches("E8Flex/Stripes/300_300")
    report = ds.validate_scene_data_flow("0")

    assert len(matches) > 1
    assert ds.classify_scene_reference("E8Flex/Stripes/300_300") == "ambiguous_name"
    assert any(message.code == "multiscale_invalid" for message in report.errors)


def test_external_vsi_subset_repair_keeps_source_dataset_unchanged(external_vsi_path, external_vsi_subset):
    source_attrs_before = json.loads((external_vsi_path / "0" / ".zattrs").read_text(encoding="utf-8"))

    ds = open_dataset(external_vsi_subset, mode="a")
    table, report = ds.ensure_plane_table("0")
    repair = ds.repair_axis_metadata("0", persist=True)

    source_attrs_after = json.loads((external_vsi_path / "0" / ".zattrs").read_text(encoding="utf-8"))

    assert report.persisted is True
    assert repair.persisted is True
    assert len(table["positioners_z"]) == 3400
    assert repair.axis_states["z"].value == 0.75
    assert source_attrs_before == source_attrs_after
    assert source_attrs_after["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 1.0
    assert source_attrs_after["omero"]["channels"][0]["window"]["max"] == 22800.0


def test_external_label_timepoint_regression_on_original_fixture(external_labels_subset):
    ds = open_dataset(external_labels_subset, mode="a")
    source = ds.level_ref("0", 0)
    source_chunks = tuple(int(chunk) for chunk in ds.read_level_zarr("0", 0).chunks)
    frame_shape = (1, 1, *source.shape[2:])
    frame_chunks = (1, 1, max(1, source_chunks[2]), min(source.shape[3], 1024), min(source.shape[4], 1024))

    def pattern(z_range: tuple[int, int], y_range: tuple[int, int], x_range: tuple[int, int], value: int):
        template = da.zeros(frame_shape, chunks=frame_chunks, dtype=np.uint16)

        def _fill(block, block_info=None):
            arr = np.zeros(block.shape, dtype=np.uint16)
            location = block_info[None]["array-location"]
            z_loc, y_loc, x_loc = location[2], location[3], location[4]
            z_start = max(z_range[0], z_loc[0])
            z_stop = min(z_range[1], z_loc[1])
            y_start = max(y_range[0], y_loc[0])
            y_stop = min(y_range[1], y_loc[1])
            x_start = max(x_range[0], x_loc[0])
            x_stop = min(x_range[1], x_loc[1])
            if z_start < z_stop and y_start < y_stop and x_start < x_stop:
                arr[
                    :,
                    :,
                    z_start - z_loc[0] : z_stop - z_loc[0],
                    y_start - y_loc[0] : y_stop - y_loc[0],
                    x_start - x_loc[0] : x_stop - x_loc[0],
                ] = value
            return arr

        return template.map_blocks(_fill, dtype=np.uint16)

    first = pattern((0, 2), (0, 32), (0, 32), 5)
    second = pattern((10, 12), (64, 96), (96, 128), 9)

    report_a = ds.write_label_timepoint("0", "external_segmentation", first, timepoint=2, attrs={"source_channel": 0})
    report_b = ds.write_label_timepoint("0", "external_segmentation", second, timepoint=7)

    reopened = open_dataset(external_labels_subset)
    level0 = reopened.root["0"]["labels"]["external_segmentation"]["0"]
    metadata = reopened.read_label_metadata("0", "external_segmentation")

    assert report_a.initialized is True
    assert report_b.initialized is False
    assert report_a.written_timepoint == 2
    assert report_b.written_timepoint == 7
    assert metadata.microio["written_timepoints"] == [2, 7]
    assert int(level0[2, 0, 0, 0, 0]) == 5
    assert int(level0[7, 0, 10, 80, 100]) == 9
