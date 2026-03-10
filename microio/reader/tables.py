"""Reader helpers for microio table groups."""

from __future__ import annotations

import numpy as np

from microio.common.models import DatasetHandle


def read_table(ds: DatasetHandle, scene_id: str, table_name: str) -> dict[str, np.ndarray]:
    """Read a table group into memory and return raw arrays."""
    table = ds.root[scene_id]["tables"][table_name]
    out: dict[str, np.ndarray] = {}
    for key, arr in table.arrays():
        out[key] = arr[:]
    return out


def read_table_metadata(ds: DatasetHandle, scene_id: str, table_name: str) -> dict:
    """Return table attrs as a plain dictionary."""
    table = ds.root[scene_id]["tables"][table_name]
    return table.attrs.asdict()


def build_axis_positions(
    ds: DatasetHandle,
    scene_id: str,
    axis: str,
    *,
    table_name: str = "axes_trajectory",
    spacing: float | None = None,
    origin: float = 0.0,
    order=None,
) -> np.ndarray:
    """Construct artificial positions for one axis using order and spacing."""
    axis = axis.lower()
    if axis not in {"t", "c", "z", "y", "x"}:
        raise ValueError(f"Unsupported axis: {axis}")

    table = read_table(ds, scene_id, table_name)
    if order is None:
        order_col = f"the_{axis}"
        if order_col in table:
            order_values = np.asarray(table[order_col], dtype=np.float64)
        elif axis in {"x", "y"}:
            order_values = np.zeros(len(next(iter(table.values()))), dtype=np.float64)
        else:
            raise ValueError(f"No default ordering column available for axis {axis!r}")
    else:
        order_values = np.asarray(order, dtype=np.float64)

    if spacing is None:
        scene_md = ds.read_scene_metadata(scene_id)
        spacing = float(scene_md.get("microio", {}).get("axis_resolution", {}).get(axis, {}).get("value", 1.0))

    return origin + (order_values * float(spacing))
