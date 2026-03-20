from __future__ import annotations

from pathlib import Path
import shutil
import uuid

import pytest

from microio.reader.fixtures import clone_scene_subset


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = REPO_ROOT.parent / "data_in" / "zarr"
SCRATCH_ROOT = REPO_ROOT.parent / "data_out" / "test_fixtures"


def _fresh_dir(prefix: str) -> Path:
    SCRATCH_ROOT.mkdir(parents=True, exist_ok=True)
    path = SCRATCH_ROOT / f"{prefix}_{uuid.uuid4().hex}.zarr"
    if path.exists():
        shutil.rmtree(path)
    return path


@pytest.fixture()
def lif_subset() -> Path:
    path = clone_scene_subset(
        DATA_ROOT / "lif_test.zarr",
        _fresh_dir("lif_subset"),
        ["14", "15", "16"],
    )
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def vsi_subset() -> Path:
    path = clone_scene_subset(
        DATA_ROOT / "vsi_test.zarr",
        _fresh_dir("vsi_subset"),
        ["0"],
    )
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)


@pytest.fixture()
def test_labels_subset() -> Path:
    path = clone_scene_subset(
        DATA_ROOT / "test_labels.zarr",
        _fresh_dir("test_labels_subset"),
        ["0"],
    )
    try:
        yield path
    finally:
        if path.exists():
            shutil.rmtree(path, ignore_errors=True)
