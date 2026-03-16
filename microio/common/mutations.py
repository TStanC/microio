"""Shared mutation helpers for enrichment writes on existing datasets."""

from __future__ import annotations

from typing import Any


def require_writable(ds) -> None:
    """Reject persistence operations on read-only dataset handles."""
    if ds.mode == "r":
        raise PermissionError("Dataset was opened read-only; reopen with mode='a' to persist changes.")


def replace_attrs(node: Any, attrs: dict[str, Any]) -> None:
    """Replace all attributes on a Zarr node atomically at the Python level."""
    node.attrs.clear()
    node.attrs.update(attrs)


def write_scene_attrs(ds, scene_id: str, attrs: dict[str, Any]) -> None:
    """Persist scene attrs and invalidate handle-local caches."""
    replace_attrs(ds.root[scene_id], attrs)
    ds.invalidate_caches(scene_id=scene_id)

