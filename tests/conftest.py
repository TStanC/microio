from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.datasets import cleanup_dataset, create_label_timepoint_dataset, create_lif_like_dataset, create_vsi_like_dataset, fresh_dataset_path
from tests.helpers.external_datasets import clone_scene_subset


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_addoption(parser) -> None:
    parser.addoption("--run-external-datasets", action="store_true", default=False, help="Run opt-in regression tests against external datasets.")
    parser.addoption("--external-lif-path", action="store", default=None, help="Path to an external LIF-derived OME-Zarr dataset.")
    parser.addoption("--external-vsi-path", action="store", default=None, help="Path to an external VSI-derived OME-Zarr dataset.")
    parser.addoption("--external-labels-path", action="store", default=None, help="Path to an external labels/timepoint OME-Zarr dataset.")


def pytest_configure(config) -> None:
    config.addinivalue_line("markers", "external_dataset: opt-in regression tests requiring explicit external dataset paths")


def pytest_collection_modifyitems(config, items) -> None:
    if config.getoption("--run-external-datasets"):
        return
    skip_external = pytest.mark.skip(reason="external dataset regression tests require --run-external-datasets")
    for item in items:
        if "external_dataset" in item.keywords:
            item.add_marker(skip_external)


@pytest.fixture()
def lif_subset() -> Path:
    path = fresh_dataset_path("lif_subset")
    create_lif_like_dataset(path, scene_ids=["14", "15", "16"])
    try:
        yield path
    finally:
        cleanup_dataset(path)


@pytest.fixture()
def lif_like_full() -> Path:
    path = fresh_dataset_path("lif_full")
    create_lif_like_dataset(path, scene_ids=["0", "14", "15", "16"], include_invalid_scene_zero=True)
    try:
        yield path
    finally:
        cleanup_dataset(path)


@pytest.fixture()
def vsi_subset() -> Path:
    path = fresh_dataset_path("vsi_subset")
    create_vsi_like_dataset(path)
    try:
        yield path
    finally:
        cleanup_dataset(path)


@pytest.fixture()
def test_labels_subset() -> Path:
    path = fresh_dataset_path("test_labels_subset")
    create_label_timepoint_dataset(path)
    try:
        yield path
    finally:
        cleanup_dataset(path)


def _external_dataset_path(request, option_name: str) -> Path:
    raw = request.config.getoption(option_name)
    if not raw:
        pytest.skip(f"{option_name} was not supplied")
    path = Path(raw)
    if not path.exists():
        pytest.skip(f"{option_name} does not exist: {path}")
    return path


@pytest.fixture()
def external_lif_path(request) -> Path:
    return _external_dataset_path(request, "--external-lif-path")


@pytest.fixture()
def external_vsi_path(request) -> Path:
    return _external_dataset_path(request, "--external-vsi-path")


@pytest.fixture()
def external_labels_path(request) -> Path:
    return _external_dataset_path(request, "--external-labels-path")


@pytest.fixture()
def external_vsi_subset(external_vsi_path: Path) -> Path:
    path = fresh_dataset_path("external_vsi_subset")
    clone_scene_subset(external_vsi_path, path, ["0"])
    try:
        yield path
    finally:
        cleanup_dataset(path)


@pytest.fixture()
def external_labels_subset(external_labels_path: Path) -> Path:
    path = fresh_dataset_path("external_labels_subset")
    clone_scene_subset(external_labels_path, path, ["0"])
    try:
        yield path
    finally:
        cleanup_dataset(path)
