from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import dask.array as da
import numpy as np
import pytest
import zarr

from microio.reader.open import open_dataset


DATA_ROOT = Path(__file__).resolve().parents[3] / "data_in" / "zarr"


def test_scene_resolution_by_id_index_and_name(vsi_subset):
    ds = open_dataset(vsi_subset)

    refs = ds.list_scene_refs()
    assert [(ref.id, ref.index, ref.name) for ref in refs] == [("0", 0, "C555")]
    assert ds.scene_ref("0").name == "C555"
    assert ds.scene_ref(0).id == "0"
    assert ds.scene_ref("C555").id == "0"
    assert ds.scene_id_to_index("0") == 0
    assert ds.scene_index_to_id(0) == "0"
    assert ds.classify_scene_reference("0") == "id"
    assert ds.classify_scene_reference("C555") == "name"
    assert ds.classify_scene_reference(0) == "index"
    assert [ref.id for ref in ds.scene_name_matches("C555")] == ["0"]


def test_duplicate_scene_name_is_explicitly_ambiguous():
    ds = open_dataset(DATA_ROOT / "lif_test.zarr")

    matches = ds.scene_name_matches("E8Flex/Stripes/300_300")
    assert len(matches) > 1
    assert ds.classify_scene_reference("E8Flex/Stripes/300_300") == "ambiguous_name"
    with pytest.raises(KeyError, match="ambiguous"):
        ds.scene_ref("E8Flex/Stripes/300_300")


def test_level_access_by_index_and_path(vsi_subset):
    ds = open_dataset(vsi_subset)

    levels = ds.list_levels("0")
    assert [level.path for level in levels] == ["0", "1", "2", "3", "4"]
    assert levels[0].shape == (100, 1, 34, 2818, 2824)
    assert levels[1].shape == (100, 1, 34, 1409, 1412)

    by_index = ds.level_ref("0", 1)
    by_path = ds.level_ref("0", "1")
    assert by_index == by_path

    arr0 = ds.read_level("0", 0)
    arr1 = ds.read_level("0", "1")
    raw = ds.read_level_zarr("0", 1)
    eager = ds.read_level_numpy("0", 1)

    assert isinstance(arr0, da.Array)
    assert isinstance(arr1, da.Array)
    assert isinstance(raw, zarr.Array)
    assert isinstance(eager, np.ndarray)
    assert arr0.shape == levels[0].shape
    assert arr1.shape == levels[1].shape
    assert eager.shape == levels[1].shape
    assert raw.shape == levels[1].shape
    assert arr1[(slice(0, 1), slice(None), slice(0, 2), slice(0, 3), slice(0, 4))].compute().shape == (
        1,
        1,
        2,
        3,
        4,
    )


def test_scene_accessor_and_data_flow_validation(vsi_subset):
    ds = open_dataset(vsi_subset)

    scene = ds.get_scene("C555")
    report = ds.validate_scene_data_flow(scene.ref.id)

    assert scene.ref.id == "0"
    assert scene.level(0).path == "0"
    assert isinstance(scene.array(0), da.Array)
    assert isinstance(scene.zarr_array(0), zarr.Array)
    assert isinstance(scene.numpy_array(0), np.ndarray)
    assert scene.array(0).shape == (100, 1, 34, 2818, 2824)
    assert report.errors == []
    assert report.warnings == []


def test_scene_metadata_cache_is_invalidated_after_persisted_repair(vsi_subset):
    ds = open_dataset(vsi_subset, mode="a")

    before = ds.read_scene_metadata("0", corrected=False)
    repair = ds.repair_axis_metadata("0", persist=True)
    after_raw = ds.read_scene_metadata("0", corrected=False)
    after_corrected = ds.read_scene_metadata("0", corrected=True)

    assert repair.persisted is True
    assert before.get("microio") is None
    assert after_raw["microio"]["repair"]["repaired_axes"]["z"]["value"] == 0.75
    assert after_raw["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 0.75
    assert after_corrected["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]["scale"][2] == 0.75


def test_scene_index_maps_to_original_ome_index_in_subset(lif_subset):
    ds = open_dataset(lif_subset)

    ref = ds.scene_ref(1)
    ome = ds.read_scene_ome_metadata(1)
    report = ds.validate_scene_data_flow(1)

    assert ref.id == "15"
    assert ref.ome_index == 15
    assert ome.name == "E8Flex/Stripes/300_300_Merged"
    assert report.errors == []


def test_scene_ome_matching_rejects_full_dataset_index_candidate_with_mismatched_name():
    dataset = _fresh_dataset_path("ome_index_name_mismatch")
    try:
        _create_named_dataset(
            dataset,
            scene_ids=["0", "1"],
            scene_names=["scene_a", "scene_b"],
            ome_names=["wrong_name", "scene_b"],
        )
        ds = open_dataset(dataset)
        with pytest.raises(KeyError, match="Could not match scene"):
            ds.read_scene_ome_metadata("0")
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_duplicate_name_subset_requires_explicit_ome_override():
    dataset = _fresh_dataset_path("ome_duplicate_subset")
    try:
        _create_named_dataset(
            dataset,
            scene_ids=["10", "20"],
            scene_names=["dup", "dup"],
            ome_names=["dup", "dup", "dup"],
            series=["10", "20"],
        )
        ds = open_dataset(dataset)
        with pytest.raises(KeyError, match="Could not match scene"):
            ds.read_scene_ome_metadata("10")
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_duplicate_name_subset_can_use_explicit_ome_override():
    dataset = _fresh_dataset_path("ome_duplicate_subset_override")
    try:
        _create_named_dataset(
            dataset,
            scene_ids=["10", "20"],
            scene_names=["dup", "dup"],
            ome_names=["dup", "dup", "dup"],
            series=["10", "20"],
        )
        ds = open_dataset(dataset, ome_scene_map={"10": 1, "20": 2})
        assert ds.scene_ref("10").ome_index == 1
        assert ds.read_scene_ome_metadata("10").index == 1
        assert ds.read_scene_ome_metadata("20").index == 2
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_duplicate_name_full_dataset_can_match_by_validated_order():
    dataset = _fresh_dataset_path("ome_duplicate_full")
    try:
        _create_named_dataset(
            dataset,
            scene_ids=["0", "1"],
            scene_names=["dup", "dup"],
            ome_names=["dup", "dup"],
        )
        ds = open_dataset(dataset)
        assert ds.scene_ref("0").ome_index == 0
        assert ds.scene_ref("1").ome_index == 1
        assert ds.read_scene_ome_metadata("0").index == 0
        assert ds.read_scene_ome_metadata("1").index == 1
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_open_dataset_rejects_invalid_ome_override():
    dataset = _fresh_dataset_path("ome_override_invalid")
    try:
        _create_named_dataset(
            dataset,
            scene_ids=["0"],
            scene_names=["scene"],
            ome_names=["scene"],
        )
        with pytest.raises(ValueError, match="out of bounds"):
            open_dataset(dataset, ome_scene_map={"0": 4})
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_open_dataset_append_mode_requires_existing_path():
    dataset = _fresh_dataset_path("missing_append_target")
    try:
        with pytest.raises(FileNotFoundError, match="does not exist"):
            open_dataset(dataset, mode="a")
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_real_lif_graph_scene_is_reported_as_invalid_not_crashing():
    ds = open_dataset(DATA_ROOT / "lif_test.zarr")

    report = ds.validate_scene_data_flow("0")
    assert any(message.code == "multiscale_invalid" for message in report.errors)


def test_missing_level_path_is_rejected():
    dataset = _fresh_dataset_path("missing_level")
    try:
        _create_minimal_dataset(dataset)
        ds = open_dataset(dataset)
        with pytest.raises(ValueError, match="missing from the Zarr group"):
            ds.list_levels("0")
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def test_data_flow_reports_ome_shape_mismatch():
    dataset = _fresh_dataset_path("shape_mismatch")
    try:
        _create_minimal_dataset(
            dataset,
            include_level_one=False,
            ome_xml=(
                '<?xml version="1.0" encoding="UTF-8"?>'
                '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
                '<Image Name="scene"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="8" SizeX="8"/></Image>'
                "</OME>"
            ),
        )
        ds = open_dataset(dataset)
        report = ds.validate_scene_data_flow("0")
        assert any(message.code == "ome_shape_mismatch" for message in report.errors)
    finally:
        if dataset.exists():
            shutil.rmtree(dataset, ignore_errors=True)


def _create_minimal_dataset(
    path: Path,
    *,
    include_level_one: bool = True,
    ome_xml: str | None = None,
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
                        {"name": "z", "type": "space"},
                        {"name": "y", "type": "space", "unit": "micrometer"},
                        {"name": "x", "type": "space", "unit": "micrometer"},
                    ],
                    "datasets": [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                        },
                        {
                            "path": "1",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 2.0, 2.0]}],
                        },
                    ]
                    if include_level_one
                    else [
                        {
                            "path": "0",
                            "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                        }
                    ],
                }
            ]
        }
    )
    scene.create_array("0", data=np.zeros((1, 1, 1, 4, 4), dtype=np.uint16))
    if include_level_one:
        # Intentionally omit the array at path "1".
        pass

    ome = root.create_group("OME")
    ome.attrs["series"] = ["0"]
    if ome_xml is None:
        ome_xml = (
            '<?xml version="1.0" encoding="UTF-8"?>'
            '<OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">'
            '<Image Name="scene"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="4" SizeX="4"/></Image>'
            "</OME>"
        )
    (path / "OME" / "METADATA.ome.xml").write_text(ome_xml, encoding="utf-8")
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_ROOT.parents[1] / "data_out" / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path


def _create_named_dataset(
    path: Path,
    *,
    scene_ids: list[str],
    scene_names: list[str],
    ome_names: list[str],
    series: list[str] | None = None,
) -> Path:
    root = zarr.open(path, mode="w")
    for scene_id, scene_name in zip(scene_ids, scene_names, strict=True):
        scene = root.create_group(scene_id)
        scene.attrs.update(
            {
                "multiscales": [
                    {
                        "name": scene_name,
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
                                "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 1.0, 1.0, 1.0]}],
                            }
                        ],
                    }
                ]
            }
        )
        scene.create_array("0", data=np.zeros((1, 1, 1, 4, 4), dtype=np.uint16))
    ome = root.create_group("OME")
    ome.attrs["series"] = list(series or scene_ids)
    images = "".join(
        f'<Image Name="{name}"><Pixels SizeT="1" SizeC="1" SizeZ="1" SizeY="4" SizeX="4"/></Image>'
        for name in ome_names
    )
    (path / "OME" / "METADATA.ome.xml").write_text(
        f'<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06">{images}</OME>',
        encoding="utf-8",
    )
    return path
