"""Access helpers for microio extension metadata blocks."""

from __future__ import annotations

from microio.common.models import DatasetHandle


def read_microio_extras(ds: DatasetHandle, scene_id: str) -> dict:
    """Return microio extras dict from scene attrs, or empty dict if absent."""
    attrs = ds.root[scene_id].attrs.asdict()
    return attrs.get("microio", {})
