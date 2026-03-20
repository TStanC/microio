"""Shared NGFF metadata helpers for mixed Zarr v2/v3 datasets."""

from __future__ import annotations

from copy import deepcopy
from typing import Any

from microio.common.constants import DEFAULT_TARGET_NGFF
from microio.common.mutations import replace_attrs


OME_ATTR_KEYS = frozenset(
    {
        "bioformats2raw.layout",
        "series",
        "multiscales",
        "omero",
        "labels",
        "image-label",
        "plate",
        "well",
    }
)


def node_zarr_format(node: Any) -> int:
    """Return the Zarr format number for a node."""
    metadata = getattr(node, "metadata", None)
    return int(getattr(metadata, "zarr_format", 2) or 2)


def flattened_attrs(node: Any) -> dict[str, Any]:
    """Return node attrs with any v3 ``ome`` namespace projected to top-level keys."""
    attrs = node.attrs.asdict()
    out = deepcopy(attrs)
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        for key, value in ome.items():
            if key == "version" or key in out:
                continue
            out[key] = deepcopy(value)
    return out


def ome_metadata(node: Any) -> dict[str, Any]:
    """Return only the semantic OME metadata for a node."""
    attrs = node.attrs.asdict()
    out: dict[str, Any] = {}
    ome = attrs.get("ome")
    if isinstance(ome, dict):
        for key, value in ome.items():
            if key == "version":
                continue
            out[key] = deepcopy(value)
    for key in OME_ATTR_KEYS:
        if key in attrs and key not in out:
            out[key] = deepcopy(attrs[key])
    return out


def non_ome_attrs(node: Any) -> dict[str, Any]:
    """Return attrs that are not part of the OME metadata namespace."""
    attrs = node.attrs.asdict()
    out: dict[str, Any] = {}
    for key, value in attrs.items():
        if key == "ome" or key in OME_ATTR_KEYS:
            continue
        out[key] = deepcopy(value)
    return out


def replace_ome_attrs(
    node: Any,
    ome: dict[str, Any],
    *,
    extra_attrs: dict[str, Any] | None = None,
    ome_version: str = DEFAULT_TARGET_NGFF,
) -> None:
    """Replace attrs on a node while serializing OME metadata for its Zarr format."""
    attrs = deepcopy(extra_attrs or {})
    if node_zarr_format(node) >= 3:
        attrs["ome"] = {"version": ome_version, **deepcopy(ome)}
    else:
        attrs.update(deepcopy(ome))
    replace_attrs(node, attrs)
