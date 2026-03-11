"""Reader helpers for microio table groups."""

from __future__ import annotations

import logging
import numpy as np

from microio.common.models import DatasetHandle


logger = logging.getLogger("microio.reader.tables")


def read_table(ds: DatasetHandle, scene_id: str, table_name: str) -> dict[str, np.ndarray]:
    """Read a complete table group into memory.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene identifier containing the table group.
    table_name:
        Table name under ``scene/tables``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from column name to eagerly loaded NumPy array.
    """
    table = ds.root[scene_id]["tables"][table_name]
    out: dict[str, np.ndarray] = {}
    for key, arr in table.arrays():
        out[key] = arr[:]
        logger.debug("Read table column %s from %s/%s with %d rows", key, scene_id, table_name, len(out[key]))
    return out


def read_table_metadata(ds: DatasetHandle, scene_id: str, table_name: str) -> dict:
    """Read table-group attributes as a plain dictionary.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene identifier containing the table.
    table_name:
        Table name under ``scene/tables``.

    Returns
    -------
    dict
        Table metadata dictionary.
    """
    logger.debug("Reading table metadata for %s/%s", scene_id, table_name)
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
    """Construct synthetic axis positions from logical indices and spacing.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene identifier containing the table.
    axis:
        Axis name, one of ``t``, ``c``, ``z``, ``y``, or ``x``.
    table_name:
        Table name under ``scene/tables`` used as the source of ordering
        columns.
    spacing:
        Optional spacing override. When omitted, spacing is read from the
        ``microio.axis_resolution`` scene metadata and defaults to ``1.0``.
    origin:
        Offset added to every generated coordinate.
    order:
        Optional explicit logical-order vector. When omitted, the function uses
        ``the_<axis>`` from the table when available.

    Returns
    -------
    numpy.ndarray
        Generated one-dimensional coordinate vector aligned to table rows.
    """
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
        logger.debug("Resolved spacing for axis %s in scene %s from metadata: %s", axis, scene_id, spacing)

    return origin + (order_values * float(spacing))
