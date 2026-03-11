"""Dataset opening helpers for OME-Zarr readers."""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.open")


def open_dataset(path: str | Path) -> DatasetHandle:
    """Open an OME-Zarr dataset in read-only mode.

    Parameters
    ----------
    path:
        Path to the root directory of an existing Zarr or OME-Zarr dataset.

    Returns
    -------
    DatasetHandle
        Thin handle exposing convenience methods for scene, metadata, table,
        and array access.
    """
    p = Path(path)
    logger.info("Opening dataset for reading: %s", p)
    root = zarr.open(p, mode="r")
    return DatasetHandle(path=p, root=root)
