"""Metadata accessors for root and scene groups."""

from __future__ import annotations

import logging

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.metadata")


def list_scenes(ds: DatasetHandle) -> list[str]:
    """List scene group names under the dataset root.

    Parameters
    ----------
    ds:
        Open dataset handle.

    Returns
    -------
    list[str]
        Sorted scene identifiers, excluding the reserved ``OME`` group.
    """
    names = []
    for key, _ in ds.root.groups():
        if key == "OME":
            continue
        names.append(key)
    logger.debug("Listed %d scene groups from %s", len(names), ds.path)
    return sorted(names)


def scene_metadata(ds: DatasetHandle, scene_id: str) -> dict:
    """Read one scene group's attributes as a plain dictionary.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene identifier under the dataset root.

    Returns
    -------
    dict
        JSON-like scene metadata dictionary.
    """
    logger.debug("Reading scene metadata for %s from %s", scene_id, ds.path)
    return ds.root[scene_id].attrs.asdict()


def root_metadata(ds: DatasetHandle) -> dict:
    """Read root-group attributes as a plain dictionary.

    Parameters
    ----------
    ds:
        Open dataset handle.

    Returns
    -------
    dict
        Root metadata dictionary.
    """
    logger.debug("Reading root metadata from %s", ds.path)
    return ds.root.attrs.asdict()
