"""Writers for microio-specific table data under scene groups."""

from __future__ import annotations

import math
import re

import numpy as np
import zarr

from microio.common.constants import AXES_TRAJECTORY_TABLE_NAME, MICROIO_TABLE_SCHEMA_VERSION
from microio.common.models import AxisResolution
from microio.common.units import normalize_unit
from .xmlparse import SceneXmlMeta


def write_axes_trajectory_table(
    scene_group: zarr.Group,
    scene: SceneXmlMeta,
    axis_res: dict[str, AxisResolution],
    *,
    size_t: int | None = None,
    size_c: int | None = None,
    size_z: int | None = None,
    original_metadata: dict[str, str] | None = None,
) -> None:
    """Write a raw-first per-plane localization table under ``scene/tables``.

    The stored row count always matches the logical plane count ``T * C * Z`` of
    the written image data. Raw OME plane positions are preferred; abstract OME
    indices are used only when no raw positions exist for an axis.
    """
    del axis_res  # Scene-level resolution is stored in scene attrs, not in the table.
    tables = scene_group.require_group("tables")
    if AXES_TRAJECTORY_TABLE_NAME in tables:
        del tables[AXES_TRAJECTORY_TABLE_NAME]
    table = tables.create_group(AXES_TRAJECTORY_TABLE_NAME)

    size_t = int(size_t if size_t is not None else scene.size_t)
    size_c = int(size_c if size_c is not None else scene.size_c)
    size_z = int(size_z if size_z is not None else scene.size_z)
    row_count = max(1, size_t * size_c * size_z)

    the_t, the_c, the_z, plane_rows = _build_plane_rows(scene, size_t=size_t, size_c=size_c, size_z=size_z)
    time_positions = _extract_original_metadata_time_positions(
        scene,
        original_metadata or {},
        size_t=size_t,
        size_c=size_c,
        size_z=size_z,
    )

    axis_x = _resolve_axis_values(
        plane_rows,
        raw_field="PositionX",
        raw_unit_field="PositionXUnit",
        raw_source="Plane.PositionX",
        abstract_field=None,
        row_count=row_count,
    )
    axis_y = _resolve_axis_values(
        plane_rows,
        raw_field="PositionY",
        raw_unit_field="PositionYUnit",
        raw_source="Plane.PositionY",
        abstract_field=None,
        row_count=row_count,
    )
    axis_z = _resolve_axis_values(
        plane_rows,
        raw_field="PositionZ",
        raw_unit_field="PositionZUnit",
        raw_source="Plane.PositionZ",
        abstract_field=the_z,
        row_count=row_count,
        abstract_source="Plane.TheZ",
    )
    axis_t = _resolve_axis_values(
        plane_rows,
        raw_field="DeltaT",
        raw_unit_field="DeltaTUnit",
        raw_source="Plane.DeltaT",
        abstract_field=the_t,
        row_count=row_count,
        abstract_source="Plane.TheT",
        override_raw=time_positions,
        override_source="OriginalMetadata.Time",
        override_confidence="low",
    )
    axis_c = _resolve_axis_values(
        plane_rows,
        raw_field=None,
        raw_unit_field=None,
        raw_source="missing",
        abstract_field=the_c,
        row_count=row_count,
        abstract_source="Plane.TheC",
    )

    data_cols = {
        "the_t": the_t.astype(np.int32),
        "the_c": the_c.astype(np.int32),
        "the_z": the_z.astype(np.int32),
        "positioners_t": axis_t["values"],
        "positioners_c": axis_c["values"],
        "positioners_z": axis_z["values"],
        "positioners_y": axis_y["values"],
        "positioners_x": axis_x["values"],
    }

    for name, arr in data_cols.items():
        table.create_array(name, data=arr, chunks=(min(len(arr), 8192),))

    table.attrs["schema"] = "microio.axes_trajectory"
    table.attrs["schema_version"] = MICROIO_TABLE_SCHEMA_VERSION
    table.attrs["row_axis_order"] = ["t", "c", "z"]
    table.attrs["shape_tcz"] = [size_t, size_c, size_z]
    table.attrs["axis_metadata"] = {
        "t": _with_logical_index(axis_t["metadata"], column="the_t", provenance="Plane.TheT"),
        "c": _with_logical_index(axis_c["metadata"], column="the_c", provenance="Plane.TheC"),
        "z": _with_logical_index(axis_z["metadata"], column="the_z", provenance="Plane.TheZ"),
        "y": axis_y["metadata"],
        "x": axis_x["metadata"],
    }


def _build_plane_rows(scene: SceneXmlMeta, *, size_t: int, size_c: int, size_z: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, str | None] | None]]:
    """Index raw OME planes into one row per logical ``(t, c, z)`` plane."""
    row_count = max(1, size_t * size_c * size_z)
    the_t = np.zeros(row_count, dtype=np.int32)
    the_c = np.zeros(row_count, dtype=np.int32)
    the_z = np.zeros(row_count, dtype=np.int32)
    plane_rows: list[dict[str, str | None] | None] = [None] * row_count

    for t in range(size_t):
        for c in range(size_c):
            for z in range(size_z):
                idx = _flat_index(t, c, z, size_c=size_c, size_z=size_z)
                the_t[idx] = t
                the_c[idx] = c
                the_z[idx] = z

    for plane in scene.planes:
        t = _safe_index(plane.get("TheT"), upper=size_t)
        c = _safe_index(plane.get("TheC"), upper=size_c)
        z = _safe_index(plane.get("TheZ"), upper=size_z)
        if t is None or c is None or z is None:
            continue
        idx = _flat_index(t, c, z, size_c=size_c, size_z=size_z)
        if plane_rows[idx] is not None:
            raise ValueError(f"Duplicate plane metadata for scene {scene.name!r} at (t={t}, c={c}, z={z})")
        plane_rows[idx] = plane

    return the_t, the_c, the_z, plane_rows


def _resolve_axis_values(
    plane_rows: list[dict[str, str | None] | None],
    *,
    raw_field: str | None,
    raw_unit_field: str | None,
    raw_source: str,
    abstract_field: np.ndarray | None,
    row_count: int,
    abstract_source: str | None = None,
    override_raw: tuple[np.ndarray, str | None] | None = None,
    override_source: str | None = None,
    override_confidence: str = "high",
) -> dict[str, object]:
    """Resolve one axis to a position vector plus per-axis metadata."""
    if override_raw is not None and override_source is not None:
        values, unit = override_raw
        return {
            "values": np.asarray(values, dtype=np.float64),
            "metadata": {
                "unit": unit,
                "provenance": override_source,
                "confidence": override_confidence,
                "warning_code": None,
                "missing_count": int(np.isnan(values).sum()),
            },
        }

    if raw_field is not None:
        raw_values = np.full(row_count, np.nan, dtype=np.float64)
        raw_units: set[str] = set()
        found_any = False
        for idx, plane in enumerate(plane_rows):
            if plane is None:
                continue
            value = _safe_float(plane.get(raw_field))
            if value is None:
                continue
            raw_values[idx] = value
            found_any = True
            if raw_unit_field is not None:
                unit = plane.get(raw_unit_field)
                if unit:
                    raw_units.add(unit)
        if found_any:
            if len(raw_units) > 1:
                raise ValueError(f"Mixed raw units for {raw_source}: {sorted(raw_units)}")
            return {
                "values": raw_values,
                "metadata": {
                    "unit": next(iter(raw_units), None),
                    "provenance": raw_source,
                    "confidence": "high",
                    "warning_code": None,
                    "missing_count": int(np.isnan(raw_values).sum()),
                },
            }

    if abstract_field is not None and abstract_source is not None:
        return {
            "values": np.asarray(abstract_field, dtype=np.float64),
            "metadata": {
                "unit": "abstract",
                "provenance": abstract_source,
                "confidence": "medium",
                "warning_code": "abstract_positions",
                "missing_count": 0,
            },
        }

    return {
        "values": np.full(row_count, np.nan, dtype=np.float64),
        "metadata": {
            "unit": None,
            "provenance": "missing",
            "confidence": "low",
            "warning_code": "missing_positions",
            "missing_count": row_count,
        },
    }


def _extract_original_metadata_time_positions(
    scene: SceneXmlMeta,
    original_metadata: dict[str, str],
    *,
    size_t: int,
    size_c: int,
    size_z: int,
) -> tuple[np.ndarray, str | None] | None:
    """Extract actual per-plane or per-time timestamps from vendor metadata if unambiguous."""
    pattern_value = re.compile(rf"^{re.escape(scene.name)} Value #(\d+)$", re.IGNORECASE)
    pattern_unit = re.compile(rf"^{re.escape(scene.name)} Units #(\d+)$", re.IGNORECASE)

    values_by_index: dict[int, str] = {}
    units_by_index: dict[int, str] = {}
    for key, value in original_metadata.items():
        mv = pattern_value.match(key)
        if mv:
            values_by_index[int(mv.group(1))] = value
            continue
        mu = pattern_unit.match(key)
        if mu:
            units_by_index[int(mu.group(1))] = value

    paired: list[tuple[int, float, str]] = []
    for idx, raw_unit in units_by_index.items():
        if idx not in values_by_index:
            continue
        normalized_unit, _ = normalize_unit(raw_unit)
        if normalized_unit not in {"second", "millisecond", "minute", "hour"}:
            continue
        raw_value = _safe_float(values_by_index[idx])
        if raw_value is None:
            continue
        paired.append((idx, raw_value, raw_unit))

    if not paired:
        return None

    paired.sort(key=lambda item: item[0])
    units = {unit for _, _, unit in paired if unit}
    if len(units) > 1:
        raise ValueError(f"Mixed OriginalMetadata time units for scene {scene.name!r}: {sorted(units)}")

    values = np.asarray([value for _, value, _ in paired], dtype=np.float64)
    row_count = max(1, size_t * size_c * size_z)
    full_row_count = max(1, scene.size_t * scene.size_c * scene.size_z)
    if len(values) == full_row_count:
        full_grid = values.reshape(scene.size_t, scene.size_z, scene.size_c)
        sliced = full_grid[:size_t, :size_z, :size_c]
        ordered = np.full(row_count, np.nan, dtype=np.float64)
        for t in range(size_t):
            for z in range(size_z):
                for c in range(size_c):
                    ordered[_flat_index(t, c, z, size_c=size_c, size_z=size_z)] = sliced[t, z, c]
        return ordered, next(iter(units), None)
    if len(values) == scene.size_t:
        expanded = np.full(row_count, np.nan, dtype=np.float64)
        for t in range(size_t):
            for z in range(size_z):
                for c in range(size_c):
                    expanded[_flat_index(t, c, z, size_c=size_c, size_z=size_z)] = values[t]
        return expanded, next(iter(units), None)
    return None


def _flat_index(t: int, c: int, z: int, *, size_c: int, size_z: int) -> int:
    """Map ``(t, c, z)`` to flattened table row index."""
    return (t * size_c * size_z) + (c * size_z) + z


def _safe_index(raw: str | None, *, upper: int) -> int | None:
    """Parse and bounds-check a plane axis index."""
    if raw is None:
        return None
    try:
        value = int(raw)
    except Exception:
        return None
    if 0 <= value < upper:
        return value
    return None


def _safe_float(raw: str | None) -> float | None:
    """Parse a float and reject non-finite values."""
    if raw is None:
        return None
    try:
        value = float(raw)
    except Exception:
        return None
    if math.isnan(value) or math.isinf(value):
        return None
    return value


def _with_logical_index(metadata: dict[str, object], *, column: str, provenance: str) -> dict[str, object]:
    """Attach logical OME plane-index metadata for axes that have ``The*`` columns."""
    out = dict(metadata)
    out["logical_index"] = {
        "column": column,
        "unit": "abstract",
        "provenance": provenance,
        "confidence": "medium",
    }
    return out
