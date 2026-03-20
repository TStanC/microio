"""Table writers for constrained scene enrichment."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import logging
from typing import Any

import numpy as np

from microio.common.constants import MICROIO_WRITER_TABLE_SCHEMA, MICROIO_WRITER_TABLE_SCHEMA_VERSION
from microio.common.models import TableWriteReport
from microio.writer.common import ensure_group_absent_or_overwrite, require_child_group, require_writeable_scene


logger = logging.getLogger("microio.writer.tables")


def write_table(
    ds,
    scene: int | str,
    name: str,
    data: Any,
    *,
    attrs: dict[str, Any] | None = None,
    overwrite: bool = False,
    append: bool = False,
    chunk_length: int | None = None,
) -> TableWriteReport:
    """Write or append one scene-local table under ``tables/<name>``.

    Parameters
    ----------
    ds:
        Open dataset handle opened in a writable mode.
    scene:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    name:
        Table name to create under the scene ``tables/`` group.
    data:
        Table payload. Supported forms are a pandas ``DataFrame`` when pandas
        is installed, a mapping of column names to one-dimensional arrays, a
        list of row dictionaries with a consistent key order, or a flat scalar
        sequence.
    attrs:
        Optional table-level metadata attributes to store on the Zarr group.
    overwrite:
        Whether to replace an existing table with the same name.
    append:
        Whether to append rows to an existing compatible table.
    chunk_length:
        Optional chunk length for the persisted column arrays.

    Returns
    -------
    TableWriteReport
        Structured summary of the created or appended table.
    """
    ref = require_writeable_scene(ds, scene)
    logger.info(
        "Writing table %s for scene %s (append=%s overwrite=%s)",
        name,
        ref.id,
        append,
        overwrite,
    )
    columns = _normalize_table_data(data)
    scene_group = ds.root[ref.id]
    tables = require_child_group(scene_group, "tables")

    if append and name in tables:
        logger.debug("Appending %d rows to existing table %s for scene %s", _row_count(columns), name, ref.id)
        report = _append_table(ref.id, tables[name], name, columns, attrs=attrs, chunk_length=chunk_length)
    else:
        if append and name not in tables:
            overwrite = False
            logger.debug("Append requested for missing table %s; creating it instead", name)
        ensure_group_absent_or_overwrite(tables, name, overwrite=overwrite)
        table = tables.create_group(name)
        _write_table_group(table, columns, attrs=attrs, chunk_length=chunk_length)
        report = TableWriteReport(
            scene_id=ref.id,
            table_name=name,
            row_count=_row_count(columns),
            column_names=list(columns.keys()),
            persisted=True,
            appended=False,
        )

    ds.invalidate_caches(scene_id=ref.id)
    return report


def _normalize_table_data(data: Any) -> dict[str, np.ndarray]:
    """Normalize supported table inputs into a column mapping."""
    if _is_pandas_dataframe(data):
        logger.debug("Normalizing pandas DataFrame input with %d columns", len(data.columns))
        return _normalize_mapping({str(col): data[col].to_numpy() for col in data.columns})
    if isinstance(data, Mapping):
        logger.debug("Normalizing mapping table input with %d columns", len(data))
        return _normalize_mapping(data)
    if isinstance(data, Sequence) and not isinstance(data, (str, bytes, bytearray)):
        rows = list(data)
        if not rows:
            logger.debug("Normalizing empty sequence table input")
            return {"value": np.asarray([], dtype=np.float64)}
        if all(isinstance(row, Mapping) for row in rows):
            logger.debug("Normalizing row-record table input with %d rows", len(rows))
            keys = list(rows[0].keys())
            if any(list(row.keys()) != keys for row in rows):
                raise ValueError("All row mappings must use the same ordered keys")
            return _normalize_mapping({str(key): [row[key] for row in rows] for key in keys})
        if all(not isinstance(row, (Mapping, Sequence)) or isinstance(row, (str, bytes, bytearray)) for row in rows):
            logger.debug("Normalizing flat scalar sequence with %d rows", len(rows))
            return _normalize_mapping({"value": rows})
    raise TypeError("Unsupported table input; expected DataFrame, mapping of columns, list of row dicts, or flat list")


def _normalize_mapping(mapping: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """Validate a column mapping and coerce it into 1D NumPy arrays."""
    if not mapping:
        raise ValueError("Table input must contain at least one column")
    normalized: dict[str, np.ndarray] = {}
    row_count: int | None = None
    for raw_name, values in mapping.items():
        name = str(raw_name)
        if not name:
            raise ValueError("Table columns must have non-empty names")
        arr = np.asarray(values)
        if arr.ndim != 1:
            raise ValueError(f"Column {name!r} must be one-dimensional")
        if arr.dtype == object:
            if any(isinstance(item, (Mapping, list, tuple, set)) for item in arr.tolist()):
                raise ValueError(f"Column {name!r} contains nested objects and cannot be serialized safely")
            arr = arr.astype("U")
        if row_count is None:
            row_count = len(arr)
        elif len(arr) != row_count:
            raise ValueError("All table columns must have the same length")
        normalized[name] = arr
    return normalized


def _write_table_group(table, columns: dict[str, np.ndarray], *, attrs: dict[str, Any] | None, chunk_length: int | None) -> None:
    """Persist a normalized table as one Zarr array per column."""
    rows = _row_count(columns)
    chunk = min(rows or 1, chunk_length or 8192)
    logger.debug("Creating table group %s with %d rows and chunk_length=%d", table.path, rows, chunk)
    for name, arr in columns.items():
        table.create_array(name, data=arr, chunks=(chunk,))
    metadata = {
        "schema": MICROIO_WRITER_TABLE_SCHEMA,
        "schema_version": MICROIO_WRITER_TABLE_SCHEMA_VERSION,
        "columns": list(columns.keys()),
        "n_rows": rows,
    }
    if attrs:
        metadata.update(attrs)
    table.attrs.update(metadata)


def _append_table(
    scene_id: str,
    table,
    table_name: str,
    columns: dict[str, np.ndarray],
    *,
    attrs: dict[str, Any] | None,
    chunk_length: int | None,
) -> TableWriteReport:
    """Append rows to an existing table after schema compatibility checks."""
    existing_columns = list(table.attrs.get("columns", []))
    if existing_columns != list(columns.keys()):
        raise ValueError(f"Existing table {table_name!r} has columns {existing_columns}, cannot append {list(columns.keys())}")
    if attrs:
        for key, value in attrs.items():
            existing = table.attrs.get(key)
            if existing is not None and existing != value:
                raise ValueError(f"Existing table attr {key!r}={existing!r} is incompatible with append value {value!r}")

    old_rows = int(table.attrs.get("n_rows", 0))
    new_rows = _row_count(columns)
    chunk = min(max(old_rows + new_rows, 1), chunk_length or 8192)
    logger.debug(
        "Appending %d rows to %s (old_rows=%d new_total=%d chunk_hint=%d)",
        new_rows,
        table.path,
        old_rows,
        old_rows + new_rows,
        chunk,
    )
    for name, arr in columns.items():
        target = table[name]
        if str(target.dtype) != str(arr.dtype):
            raise ValueError(f"Column {name!r} dtype mismatch: existing={target.dtype} new={arr.dtype}")
        target.resize((old_rows + new_rows,))
        target[old_rows:] = arr
        if getattr(target, "chunks", None) != (chunk,):
            pass

    table.attrs["n_rows"] = old_rows + new_rows
    return TableWriteReport(
        scene_id=scene_id,
        table_name=table_name,
        row_count=old_rows + new_rows,
        column_names=list(columns.keys()),
        persisted=True,
        appended=True,
    )


def _row_count(columns: dict[str, np.ndarray]) -> int:
    first = next(iter(columns.values()), None)
    return 0 if first is None else int(len(first))


def _is_pandas_dataframe(data: Any) -> bool:
    """Return whether ``data`` is a pandas DataFrame without requiring pandas."""
    try:
        import pandas as pd
    except Exception:
        return False
    return isinstance(data, pd.DataFrame)
