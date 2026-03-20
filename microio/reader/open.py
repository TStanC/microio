"""Dataset opening helpers for bioformats2raw OME-Zarr datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.open")


def open_dataset(path: str | Path, *, mode: str = "r") -> DatasetHandle:
    """Open an existing OME-Zarr dataset and return a dataset handle.

    Parameters
    ----------
    path:
        Filesystem path to an existing bioformats2raw-style OME-Zarr store.
    mode:
        Zarr access mode passed through to :func:`zarr.open`. Use ``"r"`` for
        read-only inspection and ``"a"`` when persisting repairs, tables, or
        other microio-managed enrichments.

        Example: ``open_dataset("sample.zarr", mode="a")`` is required before
        calling :meth:`DatasetHandle.repair_axis_metadata` with
        ``persist=True``.

    Returns
    -------
    DatasetHandle
        Handle exposing validated scene lookup, metadata access, image reads,
        repair helpers, and limited write-side enrichment methods.

    Raises
    ------
    FileNotFoundError
        If the requested path does not exist.
    ValueError
        If Zarr rejects the supplied mode or cannot interpret the store.

    Examples
    --------
    >>> ds = open_dataset("plate.zarr")
    >>> ds.list_scenes()
    ['0']
    >>> writable = open_dataset("plate.zarr", mode="a")
    """
    dataset_path = Path(path)
    logger.info("Opening dataset %s with mode=%s", dataset_path, mode)
    root = zarr.open(dataset_path, mode=mode)
    return DatasetHandle(path=dataset_path, root=root, mode=mode)
