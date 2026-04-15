"""Scene-axis inspection and safe metadata repair."""

from __future__ import annotations

from copy import deepcopy
import logging
import math
import statistics

import numpy as np

from microio.common.constants import AXES_ORDER, MICROIO_SCHEMA_VERSION
from microio.common.mutations import require_writable, write_scene_attrs
from microio.common.models import AxisState, RepairReport, ValidationMessage
from microio.common.units import normalize_unit
from microio.reader.metadata import multiscale_metadata, original_metadata, scene_metadata, scene_ome_metadata
from microio.reader.timing import resolve_plane_time_source, scalar_t_from_plane_source


logger = logging.getLogger("microio.reader.repair")


def inspect_axis_metadata(ds, scene_id: int | str, *, filetype: str | None = None) -> RepairReport:
    """Inspect one scene's axis metadata without mutating the store.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.

    Returns
    -------
    RepairReport
        Structured summary of the current axis state, including placeholder
        detection, warnings, and validation errors.

    Notes
    -----
    ``filetype`` is advisory. Only ``"vsi"`` currently enables additional
    format-specific timing provenance during later repair.
    """
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
        logger.debug("Scene %s has placeholder t metadata", ref.id)
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
                message=f"Scene {ref.id} has no OME/METADATA.ome.xml; OME-backed axis repair is unavailable.",
            )
        )

    _attach_existing_repair(scene_md, axis_states)
    return RepairReport(scene_id=ref.id, persisted=False, axis_states=axis_states, warnings=warnings, errors=errors)


def repair_axis_metadata(ds, scene_id: int | str, *, persist: bool = True, filetype: str | None = None) -> RepairReport:
    """Repair scene-level metadata when stronger evidence exists.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene_id:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`, for example
        ``0``, ``"0"``, or a unique multiscale scene name.
    persist:
        If ``True``, write accepted repairs back into the scene metadata. This
        requires that the dataset was opened with ``mode="a"``.

    Returns
    -------
    RepairReport
        Final axis states and any warnings or errors generated during repair.

    Notes
    -----
    ``x`` and ``y`` calibration are never invented. ``t`` repair remains
    intentionally conservative and only succeeds when OME metadata provides a
    trustworthy scalar increment. When ``omero.channels[].window`` metadata is
    present or can be synthesized safely, the display window is also repaired
    from sampled level-0 intensities while ``window.min`` and ``window.max``
    are aligned to the image dtype bounds.
    """
    ref = ds.scene_ref(scene_id)
    logger.info("Repairing axis metadata for scene %s (persist=%s filetype=%s)", ref.id, persist, filetype)
    report = inspect_axis_metadata(ds, ref.id, filetype=filetype)
    if report.errors:
        logger.info("Skipping repair for scene %s because validation reported %d errors", ref.id, len(report.errors))
        return report

    warnings = list(report.warnings)
    errors = list(report.errors)
    axis_states = dict(report.axis_states)
    repaired_axes: dict[str, AxisState] = {}
    repaired_omero: dict | None = None

    try:
        ome_scene = scene_ome_metadata(ds, ref.id)
    except FileNotFoundError:
        ome_scene = None
        logger.info("OME-backed axis repair unavailable for scene %s because OME/METADATA.ome.xml is missing", ref.id)

    if ome_scene is not None and axis_states["z"].placeholder:
        repaired_z, z_messages = _resolve_z_axis(ref.id, ome_scene)
        warnings.extend(z_messages)
        if repaired_z is not None:
            logger.info("Resolved z axis for scene %s from %s", ref.id, repaired_z.source)
            axis_states["z"] = repaired_z
            repaired_axes["z"] = repaired_z

    if ome_scene is not None and axis_states["t"].placeholder:
        repaired_t, t_messages = _resolve_t_axis(
            ref.id,
            ome_scene,
            filetype=filetype,
            original_md=original_metadata(ds) if str(filetype or "").lower() == "vsi" else None,
        )
        warnings.extend(t_messages)
        if repaired_t is not None:
            logger.info("Resolved t axis for scene %s from %s", ref.id, repaired_t.source)
            axis_states["t"] = repaired_t
            repaired_axes["t"] = repaired_t
        else:
            logger.info("Unable to resolve scalar t axis for scene %s", ref.id)

    repaired_omero, omero_messages = _resolve_channel_windows(ds, ref.id)
    warnings.extend(omero_messages)
    if repaired_omero is not None:
        logger.info("Resolved OMERO channel windows for scene %s", ref.id)

    persisted = False
    if persist and (repaired_axes or repaired_omero is not None):
        require_writable(ds)
        attrs = deepcopy(scene_metadata(ds, ref.id, corrected=False))
        if repaired_axes:
            _apply_scene_axis_repairs(attrs, repaired_axes, ref.id)
        if repaired_omero is not None:
            _apply_scene_channel_repairs(attrs, repaired_omero)
        microio = dict(attrs.get("microio", {}))
        repair_block = dict(microio.get("repair", {}))
        repair_block["schema_version"] = MICROIO_SCHEMA_VERSION
        if repaired_axes:
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
        elif "repaired_axes" not in repair_block:
            repair_block["repaired_axes"] = {}
        repair_block["filetype"] = str(filetype) if filetype else "generic"
        microio["repair"] = repair_block
        attrs["microio"] = microio
        write_scene_attrs(ds, ref.id, attrs)
        persisted = True
        logger.info(
            "Persisted repaired metadata for scene %s: axes=%s channels=%s",
            ref.id,
            sorted(repaired_axes),
            repaired_omero is not None,
        )
    elif persist:
        logger.debug("No repairable metadata changes were found for scene %s", ref.id)

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


def _apply_scene_channel_repairs(attrs: dict, repaired_omero: dict) -> None:
    """Apply repaired ``omero`` channel metadata to a scene attrs payload."""
    attrs["omero"] = deepcopy(repaired_omero)


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


def _resolve_t_axis(
    scene_id: str,
    ome_scene,
    *,
    filetype: str | None,
    original_md: dict[str, str] | None,
) -> tuple[AxisState | None, list[ValidationMessage]]:
    """Resolve a scalar t spacing from unambiguous OME timing metadata."""
    messages: list[ValidationMessage] = []
    unit, warn = normalize_unit(ome_scene.time_increment_unit)
    if ome_scene.time_increment is not None and ome_scene.time_increment > 0 and unit not in {None, "unknown"}:
        logger.debug("Using Pixels.TimeIncrement for scalar t repair in scene %s", scene_id)
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

    plane_source, source_messages = resolve_plane_time_source(ome_scene, filetype=filetype, original_metadata=original_md)
    messages.extend(source_messages)
    if plane_source is not None:
        logger.debug("Attempting scalar t repair for scene %s from %s", scene_id, plane_source.source)
        state, plane_messages = scalar_t_from_plane_source(scene_id, plane_source)
        messages.extend(plane_messages)
        return state, messages

    messages.append(
        ValidationMessage(
            level="warning",
            code="t_not_repaired",
            message=f"Scene {scene_id} t metadata stays unresolved; no complete trusted per-plane time source was found.",
        )
    )
    return None, messages


def _resolve_channel_windows(ds, scene_id: str) -> tuple[dict | None, list[ValidationMessage]]:
    """Resolve repaired ``omero.channels[].window`` metadata from level-0 data."""
    messages: list[ValidationMessage] = []
    attrs = scene_metadata(ds, scene_id, corrected=False)
    array = ds.read_level_zarr(scene_id, 0)
    dtype = np.dtype(array.dtype)
    dtype_bounds = _dtype_window_bounds(dtype)
    if dtype_bounds is None:
        messages.append(
            ValidationMessage(
                level="warning",
                code="omero_window_dtype_unsupported",
                message=f"Scene {scene_id} dtype {dtype} does not support OMERO window bounds; channel repair skipped.",
            )
        )
        return None, messages

    channel_count = int(array.shape[1])
    sampled_windows = _sample_channel_windows(array)
    current_omero = deepcopy(attrs.get("omero")) if isinstance(attrs.get("omero"), dict) else {}
    current_channels = current_omero.get("channels")
    existing_channels = (
        [deepcopy(item) for item in current_channels]
        if isinstance(current_channels, list)
        and len(current_channels) == channel_count
        and all(isinstance(item, dict) for item in current_channels)
        else None
    )
    if existing_channels is None and current_channels is not None:
        messages.append(
            ValidationMessage(
                level="warning",
                code="omero_channels_rebuilt",
                message=(
                    f"Scene {scene_id} has unusable omero channel metadata; "
                    f"rebuilding {channel_count} channel entries from image metadata."
                ),
            )
        )
    channels = existing_channels or [_default_channel_metadata(index, channel_count) for index in range(channel_count)]

    dtype_min, dtype_max = dtype_bounds
    for index, bounds in enumerate(sampled_windows):
        start = min(max(bounds[0], dtype_min), dtype_max)
        end = min(max(bounds[1], start), dtype_max)
        channel = channels[index]
        channel.setdefault("active", True)
        channel.setdefault("coefficient", 1)
        channel.setdefault("family", "linear")
        channel.setdefault("inverted", False)
        channel["label"] = str(channel.get("label", ""))
        color = str(channel.get("color") or _default_channel_color(index, channel_count)).upper()
        channel["color"] = color if _is_hex_rgb(color) else _default_channel_color(index, channel_count)
        channel["window"] = {
            "min": float(dtype_min),
            "max": float(dtype_max),
            "start": float(start),
            "end": float(end),
        }

    repaired = deepcopy(current_omero)
    repaired["channels"] = channels
    if not isinstance(repaired.get("rdefs"), dict):
        repaired["rdefs"] = _default_rdefs(channel_count)
    return repaired if repaired != attrs.get("omero") else None, messages


def _sample_channel_windows(array) -> list[tuple[float, float]]:
    """Sample level-0 image data with a bounded set of chunk reads per channel."""
    shape = tuple(int(dim) for dim in array.shape)
    chunks = _chunk_lengths(array)
    channel_count = shape[1]
    sampled: list[tuple[float, float]] = []
    patch_origins = _sample_patch_origins(shape, chunks, max_samples=24)
    for channel in range(channel_count):
        minimum = math.inf
        maximum = -math.inf
        for t_start, z_start, y_start, x_start in patch_origins:
            patch = np.asarray(
                array[
                    slice(t_start, min(t_start + 1, shape[0])),
                    slice(channel, channel + 1),
                    slice(z_start, min(z_start + 1, shape[2])),
                    slice(y_start, min(y_start + _sample_patch_size(chunks[3], shape[3] - y_start), shape[3])),
                    slice(x_start, min(x_start + _sample_patch_size(chunks[4], shape[4] - x_start), shape[4])),
                ]
            )
            if patch.size == 0:
                continue
            minimum = min(minimum, float(np.min(patch)))
            maximum = max(maximum, float(np.max(patch)))
        if not math.isfinite(minimum) or not math.isfinite(maximum):
            minimum = 0.0
            maximum = 0.0
        sampled.append((minimum, maximum))
    return sampled


def _dtype_window_bounds(dtype: np.dtype) -> tuple[float, float] | None:
    """Return exact dtype bounds for OMERO windows."""
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return float(info.min), float(info.max)
    if np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        return float(info.min), float(info.max)
    return None


def _chunk_lengths(array) -> tuple[int, ...]:
    """Normalize Zarr chunk metadata to one integer chunk length per axis."""
    chunks = getattr(array, "chunks", None)
    if chunks is None:
        return tuple(int(dim) for dim in array.shape)
    normalized: list[int] = []
    for axis_chunks, size in zip(chunks, array.shape, strict=True):
        if isinstance(axis_chunks, int):
            normalized.append(int(axis_chunks))
        elif isinstance(axis_chunks, tuple) and axis_chunks:
            normalized.append(int(axis_chunks[0]))
        else:
            normalized.append(int(size))
    return tuple(normalized)


def _sample_chunk_starts(size: int, chunk: int, *, max_chunks: int) -> list[int]:
    """Choose deterministic chunk starts spanning an axis."""
    if size <= 0:
        return [0]
    chunk = max(1, int(chunk))
    count = max(1, math.ceil(size / chunk))
    if count <= max_chunks:
        indices = range(count)
    else:
        indices = sorted({int(round(value)) for value in np.linspace(0, count - 1, num=max_chunks)})
    return [min(index * chunk, max(size - 1, 0)) for index in indices]


def _sample_patch_origins(shape: tuple[int, ...], chunks: tuple[int, ...], *, max_samples: int) -> list[tuple[int, int, int, int]]:
    """Return deterministic chunk origins spanning t/z and a small set of y/x locations."""
    t_starts = _sample_chunk_starts(shape[0], chunks[0], max_chunks=4)
    z_starts = _sample_chunk_starts(shape[2], chunks[2], max_chunks=4)
    y_starts = _sample_chunk_starts(shape[3], chunks[3], max_chunks=2)
    x_starts = _sample_chunk_starts(shape[4], chunks[4], max_chunks=2)

    origins: list[tuple[int, int, int, int]] = []
    total = max(1, int(max_samples))
    for index in range(total):
        t_start = t_starts[index % len(t_starts)]
        z_start = z_starts[(index // len(t_starts)) % len(z_starts)]
        y_start = y_starts[(index // max(1, len(t_starts) * len(z_starts))) % len(y_starts)]
        x_start = x_starts[(index // max(1, len(t_starts) * len(z_starts) * len(y_starts))) % len(x_starts)]
        origin = (t_start, z_start, y_start, x_start)
        if origin not in origins:
            origins.append(origin)
        if len(origins) == len(t_starts) * len(z_starts) * len(y_starts) * len(x_starts):
            break
    return origins


def _sample_patch_size(chunk: int, remaining: int) -> int:
    """Choose a small in-chunk patch size for sampled intensity reads."""
    return max(1, min(int(chunk), int(remaining), 16))


def _default_channel_metadata(index: int, channel_count: int) -> dict:
    """Build a minimal OMERO channel entry."""
    return {
        "active": True,
        "coefficient": 1,
        "color": _default_channel_color(index, channel_count),
        "family": "linear",
        "inverted": False,
        "label": "",
    }


def _default_channel_color(index: int, channel_count: int) -> str:
    """Choose deterministic fallback colors for synthesized OMERO channels."""
    palette = ["808080"] if channel_count == 1 else ["00FF00", "FF0000", "0000FF", "FFFFFF", "FF00FF", "00FFFF"]
    return palette[index % len(palette)]


def _default_rdefs(channel_count: int) -> dict:
    """Build conservative fallback rendering defaults."""
    return {"defaultT": 0, "defaultZ": 0, "model": "greyscale" if channel_count == 1 else "color"}


def _is_hex_rgb(value: str) -> bool:
    """Return whether ``value`` is a 6-digit hexadecimal RGB string."""
    return len(value) == 6 and all(char in "0123456789ABCDEF" for char in value)


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
