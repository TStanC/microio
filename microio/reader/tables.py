"""Plane-table loading and generation for existing OME-Zarr scenes."""

from __future__ import annotations

import logging
import math

import numpy as np

from microio.common.constants import AXES_TRAJECTORY_TABLE_NAME, MICROIO_TABLE_SCHEMA_VERSION
from microio.common.mutations import require_writable
from microio.common.models import PlaneTableReport, TableReadResult, ValidationMessage
from microio.common.units import normalize_unit
from microio.reader.metadata import original_metadata, scene_ome_metadata
from microio.reader.timing import resolve_plane_time_source


logger = logging.getLogger("microio.reader.tables")


def list_tables(ds, scene_id: int | str) -> list[str]:
    """List persisted scene-local table names.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.

    Returns
    -------
    list[str]
        Table names under the scene-local ``tables/`` group, or an empty list
        when the scene has no tables group.
    """
    ref = ds.scene_ref(scene_id)
    logger.debug("Listing tables for scene %s", ref.id)
    tables = ds.root[ref.id].get("tables")
    if tables is None:
        return []
    return [str(name) for name in tables.keys()]


def load_table(ds, scene_id: int | str, table_name: str = AXES_TRAJECTORY_TABLE_NAME) -> dict[str, np.ndarray]:
    """Load one persisted scene-local table into eager NumPy column arrays.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    table_name:
        Name of the scene-local table under ``tables/``.

    Returns
    -------
    dict[str, numpy.ndarray]
        Mapping from column name to eager one-dimensional NumPy arrays. The
        default table includes ``the_t``, ``the_c``, ``the_z``,
        ``positioners_t``, ``positioners_z``, ``positioners_y``, and
        ``positioners_x``.
    """
    ref = ds.scene_ref(scene_id)
    logger.debug("Loading table %s for scene %s", table_name, ref.id)
    table = ds.root[ref.id]["tables"][table_name]
    return {key: arr[:] for key, arr in table.arrays()}


def read_table_metadata(ds, scene_id: int | str, table_name: str = AXES_TRAJECTORY_TABLE_NAME) -> dict:
    """Read metadata attributes for one persisted table.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    table_name:
        Name of the scene-local table under ``tables/``.

    Returns
    -------
    dict
        Table attributes stored on the scene-local Zarr group.
    """
    ref = ds.scene_ref(scene_id)
    logger.debug("Reading table metadata for %s in scene %s", table_name, ref.id)
    return ds.root[ref.id]["tables"][table_name].attrs.asdict()


def read_table(ds, scene_id: int | str, table_name: str = AXES_TRAJECTORY_TABLE_NAME) -> TableReadResult:
    """Load one persisted scene-local table together with logical user attrs.

    This reader complements :func:`microio.writer.tables.write_table` by
    returning both the eager table columns and the ``table_attrs`` block that
    corresponds to the writer ``attrs=...`` payload.
    """
    ref = ds.scene_ref(scene_id)
    logger.debug("Reading table %s with logical attrs for scene %s", table_name, ref.id)
    data = load_table(ds, ref.id, table_name=table_name)
    attrs = read_table_metadata(ds, ref.id, table_name=table_name)
    table_attrs = _table_user_attrs(attrs)
    return TableReadResult(
        scene_id=ref.id,
        table_name=str(table_name),
        data=data,
        attrs=attrs,
        table_attrs=table_attrs,
        column_names=list(data.keys()),
        row_count=len(next(iter(data.values()))) if data else 0,
    )


def build_plane_table(
    ds,
    scene_id: int | str,
    *,
    table_name: str = AXES_TRAJECTORY_TABLE_NAME,
    persist: bool = False,
    filetype: str | None = None,
):
    """Build a per-plane trajectory table from OME-XML plane metadata.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    table_name:
        Name to use under the scene ``tables/`` group when persisting.
    persist:
        If ``True``, store the generated table in the dataset. Existing tables
        with the same name are replaced.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], PlaneTableReport]
        Generated table columns and the associated build report. Rows are
        ordered by the flattened ``(t, c, z)`` traversal used by
        :func:`_flat_index`.

    Notes
    -----
    Missing plane values are represented as ``NaN`` in the floating-point
    positioner columns instead of removing rows.
    """
    ref = ds.scene_ref(scene_id)
    logger.info("Building plane table %s for scene %s (persist=%s filetype=%s)", table_name, ref.id, persist, filetype)
    ome_scene = scene_ome_metadata(ds, ref.id)
    row_count = max(1, ome_scene.size_t * ome_scene.size_c * ome_scene.size_z)
    warnings: list[ValidationMessage] = []
    expected_rows = ome_scene.size_t * ome_scene.size_c * ome_scene.size_z
    observed_rows = len(ome_scene.planes)
    if observed_rows != expected_rows:
        relation = "missing" if observed_rows < expected_rows else "extra"
        warnings.append(
            ValidationMessage(
                level="warning",
                code="plane_count_mismatch",
                message=(
                    f"Scene {ref.id} expected {expected_rows} planes but observed {observed_rows} "
                    f"({relation} plane metadata)."
                ),
            )
        )

    # Table vectors scale with plane count, not with image volume, so NumPy is
    # still the appropriate representation here.
    the_t = np.zeros(row_count, dtype=np.int32)
    the_c = np.zeros(row_count, dtype=np.int32)
    the_z = np.zeros(row_count, dtype=np.int32)
    positioners_t = np.full(row_count, np.nan, dtype=np.float64)
    positioners_z = np.full(row_count, np.nan, dtype=np.float64)
    positioners_y = np.full(row_count, np.nan, dtype=np.float64)
    positioners_x = np.full(row_count, np.nan, dtype=np.float64)
    plane_rows: list[dict[str, str | None] | None] = [None] * row_count

    for t in range(ome_scene.size_t):
        for c in range(ome_scene.size_c):
            for z in range(ome_scene.size_z):
                idx = _flat_index(t, c, z, size_c=ome_scene.size_c, size_z=ome_scene.size_z)
                the_t[idx] = t
                the_c[idx] = c
                the_z[idx] = z

    for plane in ome_scene.planes:
        t = _safe_int(plane.get("TheT"))
        c = _safe_int(plane.get("TheC"))
        z = _safe_int(plane.get("TheZ"))
        if t is None or c is None or z is None:
            continue
        if not (0 <= t < ome_scene.size_t and 0 <= c < ome_scene.size_c and 0 <= z < ome_scene.size_z):
            warnings.append(
                ValidationMessage(
                    level="warning",
                    code="plane_index_out_of_bounds",
                    message=f"Scene {ref.id} has out-of-bounds plane metadata at (t={t}, c={c}, z={z}).",
                )
            )
            continue
        idx = _flat_index(t, c, z, size_c=ome_scene.size_c, size_z=ome_scene.size_z)
        if plane_rows[idx] is not None:
            logger.error("Duplicate plane metadata for scene %s at (t=%s, c=%s, z=%s)", ref.id, t, c, z)
            raise ValueError(f"Duplicate plane metadata for scene {ref.id} at (t={t}, c={c}, z={z})")
        plane_rows[idx] = plane
        positioners_z[idx] = _safe_float(plane.get("PositionZ"))
        positioners_y[idx] = _safe_float(plane.get("PositionY"))
        positioners_x[idx] = _safe_float(plane.get("PositionX"))

    time_source, time_messages = resolve_plane_time_source(
        ome_scene,
        filetype=filetype,
        original_metadata=original_metadata(ds) if str(filetype or "").lower() == "vsi" else None,
    )
    warnings.extend(time_messages)
    if time_source is not None:
        logger.info("Using %s for positioners_t in scene %s", time_source.source, ref.id)
        positioners_t = time_source.values_tcz.reshape(row_count).astype(np.float64, copy=False)
    else:
        logger.info("Leaving positioners_t unresolved for scene %s", ref.id)

    data = {
        "the_t": the_t,
        "the_c": the_c,
        "the_z": the_z,
        "positioners_t": positioners_t,
        "positioners_z": positioners_z,
        "positioners_y": positioners_y,
        "positioners_x": positioners_x,
    }
    axis_metadata = _table_axis_metadata(ome_scene, data, time_source=time_source, filetype=filetype)
    warnings.extend(_axis_table_warnings(ref.id, axis_metadata, data))
    report = PlaneTableReport(
        scene_id=ref.id,
        table_name=table_name,
        row_count=row_count,
        persisted=False,
        warnings=warnings,
    )
    if persist:
        require_writable(ds)
        _persist_table(ds, ref.id, table_name, data, ome_scene, warnings, axis_metadata=axis_metadata)
        ds.invalidate_caches(scene_id=ref.id)
        report.persisted = True
        logger.info("Persisted plane table %s for scene %s with %d rows", table_name, ref.id, row_count)
    return data, report


def ensure_plane_table(
    ds,
    scene_id: int | str,
    *,
    table_name: str = AXES_TRAJECTORY_TABLE_NAME,
    rebuild: bool = False,
    filetype: str | None = None,
):
    """Load a compatible persisted plane table or rebuild it when needed.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    table_name:
        Name of the scene-local table under ``tables/``.
    rebuild:
        If ``True``, ignore an existing compatible table and rebuild it.

    Returns
    -------
    tuple[dict[str, numpy.ndarray], PlaneTableReport]
        Loaded or generated table columns together with the action report.

    Notes
    -----
    Compatibility is currently defined by ``schema_version``. A matching stored
    table is reused unless ``rebuild=True`` is requested.
    """
    ref = ds.scene_ref(scene_id)
    scene = ds.root[ref.id]
    if not rebuild and "tables" in scene and table_name in scene["tables"]:
        metadata = scene["tables"][table_name].attrs.asdict()
        if metadata.get("schema_version") == MICROIO_TABLE_SCHEMA_VERSION and _table_matches_filetype(metadata, filetype):
            logger.debug("Reusing existing plane table %s for scene %s", table_name, ref.id)
            table = load_table(ds, ref.id, table_name=table_name)
            return table, PlaneTableReport(
                scene_id=ref.id,
                table_name=table_name,
                row_count=len(next(iter(table.values()))) if table else 0,
                persisted=False,
            )
        logger.info("Rebuilding plane table %s for scene %s because stored metadata is incompatible", table_name, ref.id)
    return build_plane_table(ds, ref.id, table_name=table_name, persist=True, filetype=filetype)


def _persist_table(
    ds,
    scene_id: str,
    table_name: str,
    data: dict[str, np.ndarray],
    ome_scene,
    warnings: list[ValidationMessage],
    *,
    axis_metadata: dict[str, dict[str, object]],
) -> None:
    """Persist a normalized plane table into the scene ``tables`` group.

    The stored attrs include schema information, the source ``(t, c, z)``
    shape, validation warnings, and summarized unit metadata for the positioner
    columns.
    """
    scene = ds.root[scene_id]
    tables = scene.require_group("tables")
    if table_name in tables:
        logger.info("Replacing existing table %s for scene %s during persistence", table_name, scene_id)
        del tables[table_name]
    table = tables.create_group(table_name)
    for name, arr in data.items():
        _write_array(table, name, arr)

    table.attrs["schema"] = "microio.axes_trajectory"
    table.attrs["schema_version"] = MICROIO_TABLE_SCHEMA_VERSION
    table.attrs["row_axis_order"] = ["t", "c", "z"]
    table.attrs["shape_tcz"] = [ome_scene.size_t, ome_scene.size_c, ome_scene.size_z]
    table.attrs["validation"] = [message.__dict__ for message in warnings]
    table.attrs["axis_metadata"] = axis_metadata


def _table_matches_filetype(metadata: dict[str, object], filetype: str | None) -> bool:
    requested = str(filetype or "generic").lower()
    if requested != "vsi":
        return True
    t_metadata = metadata.get("axis_metadata", {}).get("t", {}) if isinstance(metadata.get("axis_metadata"), dict) else {}
    if not isinstance(t_metadata, dict):
        logger.debug("Stored table metadata has no usable t axis block for VSI reuse")
        return False
    source = str(t_metadata.get("source") or "")
    stored_filetype = str(t_metadata.get("filetype") or "generic").lower()
    logger.debug(
        "Comparing stored table filetype=%s source=%s against requested filetype=%s",
        stored_filetype,
        source,
        requested,
    )
    return stored_filetype == "vsi" or source == "Plane.DeltaT"


def _table_user_attrs(attrs: dict[str, object]) -> dict[str, object]:
    """Return logical user attrs for symmetry with ``write_table(..., attrs=...)``."""
    return {
        key: value
        for key, value in attrs.items()
        if key not in {"schema", "schema_version", "columns", "n_rows"}
    }


def _table_axis_metadata(ome_scene, data: dict[str, np.ndarray], *, time_source, filetype: str | None) -> dict[str, dict[str, object]]:
    """Build enriched table axis metadata used for attrs and validation."""
    expected_count = len(data["positioners_t"])
    t_metadata = _axis_metadata("DeltaT", ome_scene.planes, "DeltaTUnit", source=None, filetype=filetype)
    if time_source is not None:
        t_metadata["unit"] = time_source.unit
        t_metadata["raw_unit"] = time_source.raw_unit
        t_metadata["warning_code"] = time_source.warning_code
        t_metadata["missing_count"] = 0
        t_metadata["source"] = time_source.source
        t_metadata["resolved"] = True
    else:
        t_metadata["missing_count"] = expected_count
    z_metadata = _axis_metadata("PositionZ", ome_scene.planes, "PositionZUnit", source="Plane.PositionZ", filetype=filetype)
    y_metadata = _axis_metadata("PositionY", ome_scene.planes, "PositionYUnit", source="Plane.PositionY", filetype=filetype)
    x_metadata = _axis_metadata("PositionX", ome_scene.planes, "PositionXUnit", source="Plane.PositionX", filetype=filetype)
    _apply_column_resolution(z_metadata, data["positioners_z"])
    _apply_column_resolution(y_metadata, data["positioners_y"])
    _apply_column_resolution(x_metadata, data["positioners_x"])
    return {
        "t": t_metadata,
        "z": z_metadata,
        "y": y_metadata,
        "x": x_metadata,
    }


def _axis_metadata(
    value_key: str,
    planes: list[dict[str, str | None]],
    unit_key: str,
    *,
    source: str | None,
    filetype: str | None,
) -> dict[str, object]:
    """Summarize unit provenance and completeness for one positioner column."""
    raw_units = sorted({plane.get(unit_key) for plane in planes if plane.get(unit_key)})
    normalized_units: set[str | None] = set()
    observed_warning_codes: set[str] = set()
    for raw_unit in raw_units:
        unit, warning_code = normalize_unit(raw_unit)
        normalized_units.add(unit)
        if warning_code is not None:
            observed_warning_codes.add(warning_code)

    if len(normalized_units) > 1:
        raw_unit = None
        unit = None
        warning_code = "unit_mixed"
    else:
        unit = next(iter(normalized_units), None)
        raw_unit = raw_units[0] if len(raw_units) == 1 else None
        if len(raw_units) > 1:
            warning_code = "unit_mixed_normalized"
        elif len(observed_warning_codes) == 1:
            warning_code = next(iter(observed_warning_codes))
        else:
            warning_code = None
    missing_count = sum(1 for plane in planes if plane.get(value_key) is None)
    observed_count = max(0, len(planes) - missing_count)
    return {
        "unit": unit,
        "raw_unit": raw_unit,
        "warning_code": warning_code,
        "missing_count": missing_count,
        "resolved": observed_count > 0,
        "source": source if observed_count > 0 else None,
        "filetype": str(filetype) if filetype else "generic",
    }


def _apply_column_resolution(metadata: dict[str, object], column: np.ndarray) -> None:
    """Align metadata provenance with actual usable values in the built column."""
    has_usable_values = bool(np.any(np.isfinite(column)))
    metadata["resolved"] = has_usable_values
    if not has_usable_values:
        metadata["source"] = None


def _axis_table_warnings(
    scene_id: str,
    axis_metadata: dict[str, dict[str, object]],
    data: dict[str, np.ndarray],
) -> list[ValidationMessage]:
    """Derive metadata-driven warnings for table-backed axes."""
    warnings: list[ValidationMessage] = []
    for axis, column_name in {
        "t": "positioners_t",
        "z": "positioners_z",
        "y": "positioners_y",
        "x": "positioners_x",
    }.items():
        metadata = axis_metadata.get(axis, {})
        column = data[column_name]
        warnings.extend(_axis_warning_messages(scene_id, axis, column_name, metadata, column))
    return warnings


def _axis_warning_messages(
    scene_id: str,
    axis: str,
    column_name: str,
    metadata: dict[str, object],
    column: np.ndarray,
) -> list[ValidationMessage]:
    """Convert one axis metadata block plus populated column data into warnings."""
    warnings: list[ValidationMessage] = []
    source = metadata.get("source")
    unit = metadata.get("unit")
    raw_unit = metadata.get("raw_unit")
    warning_code = metadata.get("warning_code")
    missing_count = int(metadata.get("missing_count") or 0)
    all_missing = bool(np.all(~np.isfinite(column)))
    any_missing = bool(np.any(~np.isfinite(column)))

    if not source:
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_no_table_source",
                message=f"Scene {scene_id} has no usable per-plane source metadata for table axis {axis}.",
            )
        )
    if all_missing:
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_no_table_values",
                message=f"Scene {scene_id} has no usable per-plane {axis} values for table column {column_name}.",
            )
        )
    if raw_unit is not None and warning_code == "unit_unknown":
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_unit_unresolved",
                message=f"Scene {scene_id} table axis {axis} uses raw unit {raw_unit!r} that could not be normalized.",
            )
        )
    if warning_code == "unit_mixed":
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_unit_mixed",
                message=f"Scene {scene_id} table axis {axis} mixes incompatible raw units across planes.",
            )
        )
    if warning_code == "unit_mixed_normalized":
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_unit_mixed_normalized",
                message=f"Scene {scene_id} table axis {axis} mixes raw units that normalize to the same canonical unit.",
            )
        )
    if unit is None:
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_unit_missing",
                message=f"Scene {scene_id} table axis {axis} has no normalized unit metadata.",
            )
        )
    if missing_count > 0 and not all_missing and any_missing:
        warnings.append(
            ValidationMessage(
                level="warning",
                code=f"{axis}_table_values_partial",
                message=f"Scene {scene_id} table axis {axis} is only partially populated ({missing_count} missing values).",
            )
        )
    return warnings


def _write_array(group, name: str, values: np.ndarray) -> None:
    try:
        group.create_array(name, data=values, chunks=(min(len(values), 8192),))
    except AttributeError:
        group.create_dataset(name, data=values, chunks=(min(len(values), 8192),))


def _flat_index(t: int, c: int, z: int, *, size_c: int, size_z: int) -> int:
    return (t * size_c * size_z) + (c * size_z) + z


def _safe_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _safe_float(raw: str | None) -> float:
    if raw is None:
        return math.nan
    try:
        return float(raw)
    except Exception:
        return math.nan
