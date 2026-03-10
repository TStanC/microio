"""Axis resolution inference with explicit precedence and fallback tracking."""

from __future__ import annotations

import statistics
import warnings

from microio.common.models import AxisResolution
from microio.common.units import normalize_unit
from microio.common.warnings import NotImplementedMetadataWarning
from .xmlparse import SceneXmlMeta


def infer_axis_resolution(scene: SceneXmlMeta, original_metadata: dict[str, str]) -> dict[str, AxisResolution]:
    """Infer x/y/z/t spacing using explicit metadata, planes, and vendor metadata."""
    out: dict[str, AxisResolution] = {}
    out["x"] = _infer_space_axis("x", scene.physical_size_x, scene.physical_size_x_unit, scene)
    out["y"] = _infer_space_axis("y", scene.physical_size_y, scene.physical_size_y_unit, scene)
    out["z"] = _infer_z(scene)
    out["t"] = _infer_t(scene, original_metadata)
    return out


def _infer_space_axis(axis: str, explicit_value: float | None, explicit_unit: str | None, scene: SceneXmlMeta) -> AxisResolution:
    """Resolve x/y from explicit Pixel physical sizes or fallback."""
    norm_unit, warning_code = normalize_unit(explicit_unit)
    if explicit_value is not None and explicit_value > 0:
        return AxisResolution(
            axis=axis,
            value=float(explicit_value),
            unit_normalized=norm_unit,
            unit_raw=explicit_unit,
            source="pixels",
            confidence="high",
            fallback=False,
            warning_code=warning_code,
        )

    warnings.warn(f"{axis}-resolution missing for scene {scene.name}; fallback 1.0", NotImplementedMetadataWarning)
    return AxisResolution(
        axis=axis,
        value=1.0,
        unit_normalized=norm_unit if norm_unit != "unknown" else None,
        unit_raw=explicit_unit,
        source="fallback",
        confidence="low",
        fallback=True,
        warning_code="resolution_missing",
    )


def _infer_z(scene: SceneXmlMeta) -> AxisResolution:
    """Resolve z spacing from explicit metadata, then plane deltas, then fallback."""
    norm_unit, warning_code = normalize_unit(scene.physical_size_z_unit)
    if scene.physical_size_z is not None and scene.physical_size_z > 0:
        return AxisResolution("z", float(scene.physical_size_z), norm_unit, scene.physical_size_z_unit, "pixels", "high", False, warning_code)

    z_positions = []
    raw_unit = None
    for pl in scene.planes:
        pz = pl.get("PositionZ")
        if pz is None:
            continue
        try:
            z_positions.append(float(pz))
            if raw_unit is None:
                raw_unit = pl.get("PositionZUnit")
        except Exception:
            continue

    if len(z_positions) >= 2:
        diffs = [abs(b - a) for a, b in zip(z_positions[:-1], z_positions[1:]) if b != a]
        if diffs:
            value = statistics.median(diffs)
            norm2, w2 = normalize_unit(raw_unit)
            return AxisResolution("z", float(value), norm2, raw_unit, "plane_delta", "medium", False, w2)

    warnings.warn(f"z-resolution missing for scene {scene.name}; fallback 1.0", NotImplementedMetadataWarning)
    return AxisResolution("z", 1.0, None, raw_unit, "fallback", "low", True, "z_missing")


def _infer_t(scene: SceneXmlMeta, original_metadata: dict[str, str]) -> AxisResolution:
    """Resolve time spacing from explicit metadata, plane deltas, vendor keys, fallback."""
    norm_unit, warning_code = normalize_unit(scene.time_increment_unit)
    if scene.time_increment is not None and scene.time_increment > 0:
        return AxisResolution("t", float(scene.time_increment), norm_unit, scene.time_increment_unit, "pixels", "high", False, warning_code)

    dts = []
    raw_unit = None
    for pl in scene.planes:
        dt = pl.get("DeltaT")
        if dt is None:
            continue
        try:
            dts.append(float(dt))
            if raw_unit is None:
                raw_unit = pl.get("DeltaTUnit")
        except Exception:
            continue

    if len(dts) >= 2:
        diffs = [abs(b - a) for a, b in zip(dts[:-1], dts[1:]) if b != a]
        if diffs:
            val = statistics.median(diffs)
            norm2, w2 = normalize_unit(raw_unit)
            return AxisResolution("t", float(val), norm2, raw_unit, "plane_delta", "medium", False, w2)

    scene_prefix = scene.name
    values = []
    unit_token = None
    for key, value in original_metadata.items():
        if not key.startswith(scene_prefix):
            continue
        if " Value #" in key:
            try:
                values.append(float(value))
            except Exception:
                pass
        elif unit_token is None and " Units #" in key:
            unit_token = value
    if len(values) >= 2:
        values.sort()
        diffs = [abs(b - a) for a, b in zip(values[:-1], values[1:]) if b != a]
        if diffs:
            val = statistics.median(diffs)
            norm3, w3 = normalize_unit(unit_token)
            return AxisResolution("t", float(val), norm3, unit_token, "original_metadata", "medium", False, w3)

    warnings.warn(f"t-resolution missing for scene {scene.name}; fallback 1.0", NotImplementedMetadataWarning)
    return AxisResolution("t", 1.0, None, raw_unit, "fallback", "low", True, "t_missing")
