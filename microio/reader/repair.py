"""Scene-axis inspection and safe metadata repair."""

from __future__ import annotations

from copy import deepcopy
import logging
import math
import statistics

from microio.common.constants import AXES_ORDER, MICROIO_SCHEMA_VERSION
from microio.common.mutations import require_writable, write_scene_attrs
from microio.common.models import AxisState, RepairReport, ValidationMessage
from microio.common.units import normalize_unit
from microio.reader.metadata import multiscale_metadata, scene_metadata, scene_ome_metadata


logger = logging.getLogger("microio.reader.repair")


def inspect_axis_metadata(ds, scene_id: int | str) -> RepairReport:
    """Validate one scene's multiscale axis metadata without mutating the store."""
    ref = ds.scene_ref(scene_id)
    logger.debug("Inspecting axis metadata for scene %s", ref.id)
    scene_md = scene_metadata(ds, ref.id, corrected=False)
    ms = multiscale_metadata(ds, ref.id, corrected=False)
    warnings: list[ValidationMessage] = []
    errors: list[ValidationMessage] = []
    axis_states: dict[str, AxisState] = {}

    axes = ms["axes"]
    datasets = ms["datasets"]
    axis_index = {axis["name"]: idx for idx, axis in enumerate(axes)}

    for axis in ("t", "c", "z", "y", "x"):
        idx = axis_index[axis]
        raw_unit = axes[idx].get("unit")
        unit, warn = normalize_unit(raw_unit)
        level_values = [float(level["coordinateTransformations"][0]["scale"][idx]) for level in datasets]
        placeholder = _is_placeholder(axis, level_values, unit)
        value = level_values[0]
        state = AxisState(
            axis=axis,
            value=value,
            unit=unit,
            raw_unit=raw_unit,
            source="zarr",
            placeholder=placeholder,
            repaired=False,
            confidence="high",
            warning_code=warn,
        )
        axis_states[axis] = state
        if axis in {"x", "y"}:
            xy_error = _validate_xy_axis(ref.id, axis, level_values, unit, raw_unit, datasets)
            if xy_error is not None:
                errors.append(xy_error)
        elif axis == "z":
            z_error = _validate_scene_wide_placeholder(axis, level_values, unit, datasets, ref.id)
            if z_error is not None:
                errors.append(z_error)
        elif axis == "t":
            t_error = _validate_scene_wide_placeholder(axis, level_values, unit, datasets, ref.id)
            if t_error is not None:
                errors.append(t_error)

    if axis_states["t"].placeholder:
        warnings.append(
            ValidationMessage(
                level="warning",
                code="t_unresolved",
                message=f"Scene {ref.id} has placeholder t metadata; automatic scalar repair is conservative.",
            )
        )

    try:
        scene_ome_metadata(ds, ref.id)
    except FileNotFoundError:
        warnings.append(
            ValidationMessage(
                level="warning",
                code="ome_xml_missing",
                message=f"Scene {ref.id} cannot be repaired because OME/METADATA.ome.xml is missing.",
            )
        )

    _attach_existing_repair(scene_md, axis_states)
    return RepairReport(scene_id=ref.id, persisted=False, axis_states=axis_states, warnings=warnings, errors=errors)


def repair_axis_metadata(ds, scene_id: int | str, *, persist: bool = True) -> RepairReport:
    """Safely repair scene-level t/z metadata when stronger evidence exists."""
    ref = ds.scene_ref(scene_id)
    logger.info("Repairing axis metadata for scene %s (persist=%s)", ref.id, persist)
    report = inspect_axis_metadata(ds, ref.id)
    if report.errors:
        logger.info("Skipping repair for scene %s because validation reported %d errors", ref.id, len(report.errors))
        return report

    warnings = list(report.warnings)
    errors = list(report.errors)
    axis_states = dict(report.axis_states)
    repaired_axes: dict[str, AxisState] = {}

    try:
        ome_scene = scene_ome_metadata(ds, ref.id)
    except FileNotFoundError:
        logger.warning("Skipping repair for scene %s because OME/METADATA.ome.xml is missing", ref.id)
        return RepairReport(
            scene_id=ref.id,
            persisted=False,
            axis_states=axis_states,
            warnings=warnings,
            errors=errors,
        )

    if axis_states["z"].placeholder:
        repaired_z, z_messages = _resolve_z_axis(ref.id, ome_scene)
        warnings.extend(z_messages)
        if repaired_z is not None:
            logger.info("Resolved z axis for scene %s from %s", ref.id, repaired_z.source)
            axis_states["z"] = repaired_z
            repaired_axes["z"] = repaired_z

    if axis_states["t"].placeholder:
        repaired_t, t_messages = _resolve_t_axis(ref.id, ome_scene)
        warnings.extend(t_messages)
        if repaired_t is not None:
            logger.info("Resolved t axis for scene %s from %s", ref.id, repaired_t.source)
            axis_states["t"] = repaired_t
            repaired_axes["t"] = repaired_t

    persisted = False
    if persist and repaired_axes:
        require_writable(ds)
        attrs = deepcopy(scene_metadata(ds, ref.id, corrected=False))
        _apply_scene_axis_repairs(attrs, repaired_axes, ref.id)
        microio = dict(attrs.get("microio", {}))
        repair_block = dict(microio.get("repair", {}))
        repair_block["schema_version"] = MICROIO_SCHEMA_VERSION
        repair_block["repaired_axes"] = {
            axis: {
                "value": state.value,
                "unit": state.unit,
                "source": state.source,
                "confidence": state.confidence,
                "warning_code": state.warning_code,
            }
            for axis, state in repaired_axes.items()
        }
        microio["repair"] = repair_block
        attrs["microio"] = microio
        write_scene_attrs(ds, ref.id, attrs)
        persisted = True
        logger.info("Persisted repaired axes for scene %s: %s", ref.id, sorted(repaired_axes))

    return RepairReport(
        scene_id=ref.id,
        persisted=persisted,
        axis_states=axis_states,
        warnings=warnings,
        errors=errors,
    )


def _apply_scene_axis_repairs(attrs: dict, repaired_axes: dict[str, AxisState], scene_id: str) -> None:
    """Apply resolved axis values to a scene attrs payload in-place."""
    ms_list = attrs.get("multiscales", [])
    if not ms_list:
        raise ValueError(f"Scene {scene_id} has no multiscales metadata to repair")
    multiscale = deepcopy(ms_list[0])
    axes = multiscale.get("axes", [])
    datasets = multiscale.get("datasets", [])
    axis_index = {axis["name"]: idx for idx, axis in enumerate(axes)}

    for axis_name, state in repaired_axes.items():
        idx = axis_index[axis_name]
        axes[idx]["unit"] = state.unit
        for dataset in datasets:
            transforms = dataset.get("coordinateTransformations")
            if not isinstance(transforms, list) or not transforms:
                raise ValueError(f"Scene {scene_id} level {dataset.get('path')} has malformed coordinate transformations")
            scale = transforms[0].get("scale")
            if not isinstance(scale, list) or len(scale) != len(AXES_ORDER):
                raise ValueError(f"Scene {scene_id} has malformed scale vector at level {dataset.get('path')}")
            scale[idx] = float(state.value)

    multiscale["axes"] = axes
    multiscale["datasets"] = datasets
    attrs["multiscales"] = [multiscale]

def _resolve_z_axis(scene_id: str, ome_scene) -> tuple[AxisState | None, list[ValidationMessage]]:
    """Resolve a scalar z spacing from OME pixel sizes or per-plane positions."""
    messages: list[ValidationMessage] = []
    unit, warn = normalize_unit(ome_scene.physical_size_z_unit)
    if ome_scene.physical_size_z is not None and ome_scene.physical_size_z > 0 and unit not in {None, "unknown"}:
        return (
            AxisState(
                axis="z",
                value=float(ome_scene.physical_size_z),
                unit=unit,
                raw_unit=ome_scene.physical_size_z_unit,
                source="Pixels.PhysicalSizeZ",
                placeholder=False,
                repaired=True,
                confidence="high",
                warning_code=warn,
            ),
            messages,
        )

    spacing, raw_unit, details = _infer_z_from_planes(ome_scene)
    messages.extend(details)
    if spacing is None:
        return None, messages
    unit, warn = normalize_unit(raw_unit)
    if unit in {None, "unknown"}:
        messages.append(
            ValidationMessage(
                level="warning",
                code="z_unit_unresolved",
                message=f"Scene {scene_id} z spacing was inferred but unit could not be normalized.",
            )
        )
        return None, messages
    return (
        AxisState(
            axis="z",
            value=spacing,
            unit=unit,
            raw_unit=raw_unit,
            source="Plane.PositionZ",
            placeholder=False,
            repaired=True,
            confidence="medium",
            warning_code=warn,
        ),
        messages,
    )


def _resolve_t_axis(scene_id: str, ome_scene) -> tuple[AxisState | None, list[ValidationMessage]]:
    """Resolve a scalar t spacing from unambiguous OME timing metadata."""
    messages: list[ValidationMessage] = []
    unit, warn = normalize_unit(ome_scene.time_increment_unit)
    if ome_scene.time_increment is not None and ome_scene.time_increment > 0 and unit not in {None, "unknown"}:
        return (
            AxisState(
                axis="t",
                value=float(ome_scene.time_increment),
                unit=unit,
                raw_unit=ome_scene.time_increment_unit,
                source="Pixels.TimeIncrement",
                placeholder=False,
                repaired=True,
                confidence="high",
                warning_code=warn,
            ),
            messages,
        )

    messages.append(
        ValidationMessage(
            level="warning",
            code="t_not_repaired",
            message=(
                f"Scene {scene_id} t metadata stays unresolved; Plane.DeltaT and generic OriginalMetadata are "
                "not collapsed to a single scalar in v1."
            ),
        )
    )
    return None, messages


def _infer_z_from_planes(ome_scene) -> tuple[float | None, str | None, list[ValidationMessage]]:
    """Infer z spacing from per-plane ``PositionZ`` values when safe to do so."""
    by_stack: dict[tuple[int, int], list[tuple[int, float, str | None]]] = {}
    for plane in ome_scene.planes:
        raw_z = plane.get("PositionZ")
        if raw_z is None:
            continue
        try:
            the_t = int(plane.get("TheT") or 0)
            the_c = int(plane.get("TheC") or 0)
            the_z = int(plane.get("TheZ") or 0)
            value = float(raw_z)
        except Exception:
            continue
        by_stack.setdefault((the_t, the_c), []).append((the_z, value, plane.get("PositionZUnit")))

    steps: list[float] = []
    raw_units: set[str] = set()
    for stack in by_stack.values():
        ordered = sorted(stack, key=lambda item: item[0])
        diffs = []
        for (_, a, _), (_, b, _) in zip(ordered[:-1], ordered[1:]):
            if a != b:
                diffs.append(abs(b - a))
        if diffs:
            steps.append(statistics.median(diffs))
            for _, _, raw_unit in ordered:
                if raw_unit:
                    raw_units.add(raw_unit)

    messages: list[ValidationMessage] = []
    if not steps:
        messages.append(
            ValidationMessage(
                level="warning",
                code="z_no_plane_positions",
                message=f"Scene {ome_scene.name} has no usable Plane.PositionZ values for scalar z repair.",
            )
        )
        return None, None, messages

    if len(raw_units) > 1:
        messages.append(
            ValidationMessage(
                level="error",
                code="z_mixed_units",
                message=f"Scene {ome_scene.name} has mixed Plane.PositionZ units: {sorted(raw_units)}.",
            )
        )
        return None, None, messages

    spread = max(steps) - min(steps)
    median_step = statistics.median(steps)
    tolerance = max(1e-6, abs(median_step) * 0.05)
    if not math.isfinite(median_step) or median_step <= 0 or spread > tolerance:
        messages.append(
            ValidationMessage(
                level="warning",
                code="z_inconsistent_plane_spacing",
                message=(
                    f"Scene {ome_scene.name} has inconsistent per-stack z spacing "
                    f"(median={median_step}, spread={spread}); automatic repair skipped."
                ),
            )
        )
        return None, next(iter(raw_units), None), messages
    return float(median_step), next(iter(raw_units), None), messages


def _validate_xy_axis(scene_id: str, axis: str, level_values: list[float], unit: str | None, raw_unit: str | None, datasets: list[dict]) -> ValidationMessage | None:
    if _is_placeholder(axis, level_values, unit):
        return ValidationMessage(
            level="error",
            code=f"{axis}_placeholder",
            message=(
                f"Scene {scene_id} level {datasets[0].get('path', '0')} axis {axis} has unit={raw_unit!r} "
                f"and scale={level_values[0]}; x/y calibration must not be invented."
            ),
        )
    base = level_values[0]
    for current in level_values[1:]:
        if current < base:
            return ValidationMessage(
                level="error",
                code=f"{axis}_multiscale_invalid",
                message=f"Scene {scene_id} axis {axis} has non-monotonic pyramid scaling.",
            )
    return None


def _validate_scene_wide_placeholder(axis: str, level_values: list[float], unit: str | None, datasets: list[dict], scene_id: str) -> ValidationMessage | None:
    placeholders = [_is_placeholder(axis, [value], unit) for value in level_values]
    if any(placeholders) and not all(placeholders):
        return ValidationMessage(
            level="error",
            code=f"{axis}_mixed_multiscale_placeholder",
            message=(
                f"Scene {scene_id} axis {axis} has inconsistent placeholder metadata across levels: "
                f"{[(datasets[i].get('path'), level_values[i]) for i in range(len(level_values))]}."
            ),
        )
    return None


def _is_placeholder(axis: str, level_values: list[float], unit: str | None) -> bool:
    if axis == "c":
        return False
    return unit in {None, "unknown"} and all(abs(float(value) - 1.0) < 1e-9 for value in level_values)


def _attach_existing_repair(scene_md: dict, axis_states: dict[str, AxisState]) -> None:
    repair = scene_md.get("microio", {}).get("repair", {})
    for axis, metadata in repair.get("repaired_axes", {}).items():
        if axis not in axis_states:
            continue
        axis_states[axis] = AxisState(
            axis=axis,
            value=metadata.get("value"),
            unit=metadata.get("unit"),
            raw_unit=metadata.get("unit"),
            source=metadata.get("source", "microio.repair"),
            placeholder=False,
            repaired=True,
            confidence=metadata.get("confidence", "high"),
            warning_code=metadata.get("warning_code"),
        )
