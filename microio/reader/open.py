"""Dataset opening helpers for bioformats2raw OME-Zarr datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.open")


def open_dataset(path: str | Path, *, mode: str = "r") -> DatasetHandle:
    """Open an existing OME-Zarr dataset."""
    dataset_path = Path(path)
    logger.info("Opening dataset %s with mode=%s", dataset_path, mode)
    root = zarr.open(dataset_path, mode=mode)
    return DatasetHandle(path=dataset_path, root=root, mode=mode)
