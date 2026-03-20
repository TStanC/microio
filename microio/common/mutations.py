"""Shared mutation helpers for enrichment writes on existing datasets."""

from __future__ import annotations

from typing import Any

from microio.common.ngff import OME_ATTR_KEYS, replace_ome_attrs


def require_writable(ds) -> None:
    """Reject persistence operations on read-only dataset handles."""
    if ds.mode == "r":
        raise PermissionError("Dataset was opened read-only; reopen with mode='a' to persist changes.")


def replace_attrs(node: Any, attrs: dict[str, Any]) -> None:
    """Replace all attributes on a Zarr node atomically at the Python level."""
    node.attrs.clear()
    node.attrs.update(attrs)


def write_scene_attrs(ds, scene_id: str, attrs: dict[str, Any]) -> None:
    """Persist scene attrs and invalidate handle-local caches.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Canonical scene id, for example ``"0"``.
    attrs:
        Flattened semantic scene attrs as returned by
        :func:`microio.reader.metadata.scene_metadata`. For Zarr v3 scenes the
        semantic OME keys such as ``multiscales`` and ``omero`` are written
        back under the ``ome`` namespace automatically.
    """
    node = ds.root[scene_id]
    ome = {key: attrs[key] for key in OME_ATTR_KEYS if key in attrs}
    extra = {key: value for key, value in attrs.items() if key not in OME_ATTR_KEYS and key != "ome"}
    replace_ome_attrs(node, ome, extra_attrs=extra)
    ds.invalidate_caches(scene_id=scene_id)
