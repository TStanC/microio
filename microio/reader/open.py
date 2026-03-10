"""Dataset opening helpers for OME-Zarr readers."""

from __future__ import annotations

from pathlib import Path

import zarr

from microio.common.models import DatasetHandle


def open_dataset(path: str | Path) -> DatasetHandle:
    """Open a zarr dataset in read-only mode."""
    p = Path(path)
    root = zarr.open(p, mode="r")
    return DatasetHandle(path=p, root=root)
