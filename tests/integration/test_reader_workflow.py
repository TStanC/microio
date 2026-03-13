from __future__ import annotations

import json
from pathlib import Path

from microio.reader.open import open_dataset


def test_open_subset_in_dataset_order(lif_subset):
    ds = open_dataset(lif_subset)
    assert ds.list_scenes() == ["14", "15", "16"]
    assert ds.read_scene_ome_metadata("15").name == "E8Flex/Stripes/300_300_Merged"


def test_vsi_repair_persists_z_only(vsi_subset):
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

    attrs = ds.read_scene_metadata("0")
    z_axis = next(axis for axis in attrs["multiscales"][0]["axes"] if axis["name"] == "z")
    assert z_axis["unit"] == "micrometer"
    assert attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 0.75


def test_ensure_plane_table_is_idempotent(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")
    _, first = ds.ensure_plane_table("0")
    table, second = ds.ensure_plane_table("0")

    assert first.persisted is True
    assert second.persisted is False
    assert len(table["the_z"]) == 3400


def test_source_fixture_is_not_modified(vsi_subset):
    source_attrs = json.loads(
        (Path(__file__).resolve().parents[3] / "data_in" / "zarr" / "vsi_test.zarr" / "0" / ".zattrs").read_text()
    )
    assert source_attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 1.0
