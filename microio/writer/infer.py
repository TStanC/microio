"""Axis resolution inference with explicit precedence and fallback tracking."""

from __future__ import annotations

import logging
import statistics
import warnings

from microio.common.models import AxisResolution
from microio.common.units import normalize_unit
from microio.common.warnings import NotImplementedMetadataWarning
from .xmlparse import SceneXmlMeta


logger = logging.getLogger("microio.writer.infer")


def infer_axis_resolution(scene: SceneXmlMeta, original_metadata: dict[str, str]) -> dict[str, AxisResolution]:
    """Infer per-axis sampling from OME and vendor metadata.

    Parameters
    ----------
    scene:
        Normalized scene metadata parsed from OME-XML.
    original_metadata:
        Vendor ``OriginalMetadata`` key/value mapping from the same file.

    Returns
    -------
    dict[str, AxisResolution]
        Resolution records for ``x``, ``y``, ``z``, and ``t`` describing the
        resolved value, unit, provenance, confidence, and fallback status.
    """
    logger.debug("Inferring axis resolution for scene %s", scene.name)
    out: dict[str, AxisResolution] = {}
    out["x"] = _infer_space_axis("x", scene.physical_size_x, scene.physical_size_x_unit, scene)
    out["y"] = _infer_space_axis("y", scene.physical_size_y, scene.physical_size_y_unit, scene)
    out["z"] = _infer_z(scene)
    out["t"] = _infer_t(scene, original_metadata)
    logger.debug(
        "Resolved axis metadata for scene %s: %s",
        scene.name,
        {axis: {"value": meta.value, "unit": meta.unit_normalized, "source": meta.source, "fallback": meta.fallback} for axis, meta in out.items()},
    )
    return out


def _infer_space_axis(axis: str, explicit_value: float | None, explicit_unit: str | None, scene: SceneXmlMeta) -> AxisResolution:
    """Resolve an in-plane spatial axis from explicit ``Pixels`` metadata."""
    norm_unit, warning_code = normalize_unit(explicit_unit)
    if explicit_value is not None and explicit_value > 0:
        logger.debug("Scene %s axis %s resolved from Pixels metadata", scene.name, axis)
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

    logger.warning("Scene %s axis %s missing explicit spatial resolution; using fallback 1.0", scene.name, axis)
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
    """Resolve axial spacing from OME ``Pixels`` metadata or plane positions."""
    norm_unit, warning_code = normalize_unit(scene.physical_size_z_unit)
    if scene.physical_size_z is not None and scene.physical_size_z > 0:
        logger.debug("Scene %s z spacing resolved from Pixels metadata", scene.name)
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
            logger.info("Scene %s z spacing inferred from Plane.PositionZ deltas", scene.name)
            return AxisResolution("z", float(value), norm2, raw_unit, "plane_delta", "medium", False, w2)

    logger.warning("Scene %s z spacing missing; using fallback 1.0", scene.name)
    warnings.warn(f"z-resolution missing for scene {scene.name}; fallback 1.0", NotImplementedMetadataWarning)
    return AxisResolution("z", 1.0, None, raw_unit, "fallback", "low", True, "z_missing")


def _infer_t(scene: SceneXmlMeta, original_metadata: dict[str, str]) -> AxisResolution:
    """Resolve temporal spacing from OME and vendor-specific metadata."""
    norm_unit, warning_code = normalize_unit(scene.time_increment_unit)
    if scene.time_increment is not None and scene.time_increment > 0:
        logger.debug("Scene %s time increment resolved from Pixels metadata", scene.name)
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
            logger.info("Scene %s time increment inferred from Plane.DeltaT deltas", scene.name)
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
            logger.info("Scene %s time increment inferred from OriginalMetadata", scene.name)
            return AxisResolution("t", float(val), norm3, unit_token, "original_metadata", "medium", False, w3)

    logger.warning("Scene %s time increment missing; using fallback 1.0", scene.name)
    warnings.warn(f"t-resolution missing for scene {scene.name}; fallback 1.0", NotImplementedMetadataWarning)
    return AxisResolution("t", 1.0, None, raw_unit, "fallback", "low", True, "t_missing")
