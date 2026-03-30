from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import numpy as np

from microio.reader.open import open_dataset


DATA_OUT = Path(__file__).resolve().parents[3] / "data_out"


def test_workspace_round_trip_on_vsi_subset(vsi_subset):
    workspace = _fresh_dataset_path("workspace_vsi")
    try:
        source_ds = open_dataset(vsi_subset, mode="a")
        before_labels = source_ds.list_labels("0")
        source_shape = source_ds.level_ref("0", 0).shape

        seed = np.zeros(source_shape, dtype=np.uint16)
        seed[..., :2, :2] = 3
        source_ds.write_label_image("0", "seed_label", seed)
        source_ds.create_workspace(workspace, "0", labels=["seed_label"])

        workspace_ds = open_dataset(workspace, mode="a")
        assert workspace_ds.list_labels("0") == ["seed_label"]
        assert source_ds.list_labels("0") == before_labels + ["seed_label"]

        computed = np.zeros(workspace_ds.level_ref("0", 0).shape, dtype=np.uint16)
        computed[..., 1:3, 1:3] = 7
        workspace_ds.write_label_image("0", "workspace_result", computed)
        workspace_ds.commit_workspace_labels("committed_result", workspace_label="workspace_result")

        reopened_source = open_dataset(vsi_subset)
        reopened_workspace = open_dataset(workspace)
        assert reopened_source.root["0"]["labels"]["committed_result"]["0"].shape == source_shape
        assert reopened_workspace.validate_scene_data_flow("0").errors == []
    finally:
        shutil.rmtree(workspace, ignore_errors=True)


def _fresh_dataset_path(prefix: str) -> Path:
    path = DATA_OUT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
    return path
