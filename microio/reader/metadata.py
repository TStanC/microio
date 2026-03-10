"""Metadata accessors for root and scene groups."""

from __future__ import annotations

from microio.common.models import DatasetHandle


def list_scenes(ds: DatasetHandle) -> list[str]:
    """List scene group names, excluding the reserved ``OME`` group."""
    names = []
    for key, _ in ds.root.groups():
        if key == "OME":
            continue
        names.append(key)
    return sorted(names)


def scene_metadata(ds: DatasetHandle, scene_id: str) -> dict:
    """Read scene attrs as a plain dictionary."""
    return ds.root[scene_id].attrs.asdict()


def root_metadata(ds: DatasetHandle) -> dict:
    """Read root attrs as a plain dictionary."""
    return ds.root.attrs.asdict()
