from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np
import pytest
import zarr

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_create_workspace_rechunks_and_validates_metadata():
    dataset = _fresh_dataset_path("workspace_create")
    workspace = _fresh_dataset_path("workspace_copy")
    try:
        _create_workspace_source_dataset(dataset)
        ds = open_dataset(dataset, mode="a")

        report = ds.create_workspace(workspace, "0", chunks=(1, 1, 1, 4, 4))
        workspace_ds = open_dataset(workspace)
        flow = workspace_ds.validate_scene_data_flow("0")

        assert report.workspace_path == workspace
        assert report.source_dataset_path == dataset.resolve()
        assert report.source_scene_id == "0"
        assert report.source_level == "0"
        assert report.chunks == (1, 1, 1, 4, 4)
        assert tuple(int(chunk) for chunk in workspace_ds.root["0"]["0"].chunks) == (1, 1, 1, 4, 4)
        assert flow.errors == []
        assert workspace_ds.level_ref("0", 0).shape == ds.level_ref("0", 0).shape
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def test_workspace_label_carryover_is_read_only():
    dataset = _fresh_dataset_path("workspace_labels_source")
    workspace = _fresh_dataset_path("workspace_labels_copy")
    try:
        _create_workspace_source_dataset(dataset)
        source_ds = open_dataset(dataset, mode="a")
        label_data = np.zeros(source_ds.level_ref("0", 0).shape, dtype=np.uint16)
        label_data[..., 2:6, 2:6] = 5
        source_ds.write_label_image("0", "existing", label_data)

        source_ds.create_workspace(workspace, "0", labels=["existing"])
        workspace_ds = open_dataset(workspace, mode="a")
        metadata = workspace_ds.read_label_metadata("0", "existing")

        assert workspace_ds.list_labels("0") == ["existing"]
        assert metadata.microio["workspace"]["read_only"] is True
        assert metadata.microio["workspace"]["carried_from_source"] is True
        with pytest.raises(PermissionError, match="read-only workspace label"):
            workspace_ds.write_label_image("0", "existing", label_data, overwrite=True)
        with pytest.raises(PermissionError, match="read-only source label"):
            workspace_ds.commit_workspace_labels("existing", workspace_label="existing")
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def test_workspace_commit_label_and_table_round_trip():
    dataset = _fresh_dataset_path("workspace_commit_source")
    workspace = _fresh_dataset_path("workspace_commit_copy")
    try:
        _create_workspace_source_dataset(dataset)
        source_ds = open_dataset(dataset, mode="a")
        source_ds.create_workspace(workspace, "0", chunks=(1, 1, 1, 4, 4))

        workspace_ds = open_dataset(workspace, mode="a")
        computed = np.zeros(workspace_ds.level_ref("0", 0).shape, dtype=np.uint16)
        computed[..., 1:3, 1:3] = 9
        workspace_ds.write_label_image(
            "0",
            "computed",
            computed,
            attrs={"description": "workspace result", "source_channel": 0},
            colors=[{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 9, "rgba": [255, 255, 0, 255]}],
            properties=[{"label-value": 9, "class": "nucleus"}],
        )
        workspace_ds.write_table("0", "measurements", {"label_id": [9], "area": [4.0]}, attrs={"description": "workspace table"})

        label_report = workspace_ds.commit_workspace_labels("final_labels", workspace_label="computed")
        table_report = workspace_ds.commit_workspace_table("final_table", workspace_table="measurements")

        reopened = open_dataset(dataset)
        assert label_report.persisted is True
        assert table_report.persisted is True
        assert "final_labels" in reopened.list_labels("0")
        assert reopened.root["0"]["labels"]["final_labels"]["0"][0, 0, 0, 1, 1] == 9
        final_md = reopened.read_label_metadata("0", "final_labels")
        assert final_md.label_attrs == {"description": "workspace result", "source_channel": 0}
        assert final_md.colors == [{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 9, "rgba": [255, 255, 0, 255]}]
        assert final_md.properties == [{"label-value": 9, "class": "nucleus"}]
        assert reopened.load_table("0", "final_table")["label_id"].tolist() == [9]
        assert reopened.read_table("0", "final_table").table_attrs == {"description": "workspace table"}
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def test_delete_workspace_requires_workspace_provenance():
    dataset = _fresh_dataset_path("workspace_delete_source")
    workspace = _fresh_dataset_path("workspace_delete_copy")
    try:
        _create_workspace_source_dataset(dataset)
        source_ds = open_dataset(dataset, mode="a")
        source_ds.create_workspace(workspace, "0")
        workspace_ds = open_dataset(workspace, mode="a")

        deleted = workspace_ds.delete_workspace()

        assert deleted == workspace
        assert not workspace.exists()
        with pytest.raises(ValueError, match="not a microio computation workspace"):
            source_ds.delete_workspace()
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def test_create_workspace_rejects_existing_destination_without_overwrite():
    dataset = _fresh_dataset_path("workspace_existing_source")
    workspace = _fresh_dataset_path("workspace_existing_copy")
    try:
        _create_workspace_source_dataset(dataset)
        _create_workspace_source_dataset(workspace)
        ds = open_dataset(dataset, mode="a")

        with pytest.raises(FileExistsError, match="already exists"):
            ds.create_workspace(workspace, "0")
    finally:
        shutil.rmtree(dataset, ignore_errors=True)
        shutil.rmtree(workspace, ignore_errors=True)


def _create_workspace_source_dataset(path: Path) -> Path:
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
    level0 = np.arange(2 * 1 * 2 * 8 * 8, dtype=np.uint16).reshape(2, 1, 2, 8, 8)
    scene.create_array("0", data=level0, chunks=(1, 1, 1, 8, 8), dimension_names=("t", "c", "z", "y", "x"))
    scene.create_array("1", data=level0[..., ::2, ::2], chunks=(1, 1, 1, 4, 4), dimension_names=("t", "c", "z", "y", "x"))
    ome = root.create_group("OME")
    ome.attrs["ome"] = {"version": "0.5", "series": ["0"]}
    (path / "OME" / "METADATA.ome.xml").write_text(
        '<?xml version="1.0" encoding="UTF-8"?><OME xmlns="http://www.openmicroscopy.org/Schemas/OME/2016-06"><Image ID="Image:0" Name="scene"><Pixels ID="Pixels:0" DimensionOrder="XYZCT" Type="uint16" SizeT="2" SizeC="1" SizeZ="2" SizeY="8" SizeX="8"/></Image></OME>',
        encoding="utf-8",
    )
    return path


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
