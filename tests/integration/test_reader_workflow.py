from __future__ import annotations

import numpy as np

from microio.reader.open import open_dataset


def test_open_subset_in_dataset_order(lif_subset):
    ds = open_dataset(lif_subset)
    assert ds.list_scenes() == ["14", "15", "16"]
    assert ds.read_scene_ome_metadata("15").name == "E8Flex/Stripes/300_300_Merged"


def test_vsi_repair_persists_z_and_dtype_window_bounds(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")
    table, table_report = ds.ensure_plane_table("0")
    repair = ds.repair_axis_metadata("0", persist=True)

    assert table_report.persisted
    assert len(table["positioners_z"]) == 3400
    assert repair.persisted
    assert repair.axis_states["z"].repaired is True
    assert repair.axis_states["z"].value == 0.75
    assert repair.axis_states["t"].placeholder is True
    assert repair.axis_states["t"].repaired is False

    corrected = ds.read_scene_metadata("0")
    raw = ds.read_scene_metadata("0", corrected=False)
    z_axis = next(axis for axis in corrected["multiscales"][0]["axes"] if axis["name"] == "z")
    raw_z_axis = next(axis for axis in raw["multiscales"][0]["axes"] if axis["name"] == "z")
    window = raw["omero"]["channels"][0]["window"]
    assert z_axis["unit"] == "micrometer"
    assert corrected["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 0.75
    assert raw_z_axis["unit"] == "micrometer"
    assert raw["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 0.75
    assert raw["microio"]["repair"]["repaired_axes"]["z"]["value"] == 0.75
    assert window["min"] == 0.0
    assert window["max"] == 65535.0
    assert 0.0 <= window["start"] <= window["end"] <= 65535.0


def test_ensure_plane_table_is_idempotent(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")
    _, first = ds.ensure_plane_table("0")
    table, second = ds.ensure_plane_table("0")

    assert first.persisted is True
    assert second.persisted is False
    assert len(table["the_z"]) == 3400


def test_vsi_filetype_gates_numbered_time_metadata(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")

    generic_table, _ = ds.build_plane_table("0", persist=False)
    vsi_table, _ = ds.build_plane_table("0", persist=False, filetype="vsi")

    assert np.isnan(generic_table["positioners_t"]).all()
    assert np.isfinite(vsi_table["positioners_t"]).all()


def test_lif_repair_updates_channel_windows_to_dtype_bounds(lif_subset):
    ds = open_dataset(lif_subset, mode="a")
    repair = ds.repair_axis_metadata("15", persist=True)
    channels = ds.read_scene_metadata("15", corrected=False)["omero"]["channels"]

    assert repair.persisted is True
    assert len(channels) == 3
    assert [channel["window"]["min"] for channel in channels] == [0.0, 0.0, 0.0]
    assert [channel["window"]["max"] for channel in channels] == [255.0, 255.0, 255.0]
    assert all(0.0 <= channel["window"]["start"] <= channel["window"]["end"] <= 255.0 for channel in channels)
