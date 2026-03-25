"""Shared helpers for choosing a single per-plane time provenance."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import math
import re
from typing import Mapping

import numpy as np

from microio.common.models import AxisState, ValidationMessage
from microio.common.units import normalize_unit


logger = logging.getLogger("microio.reader.timing")


@dataclass(frozen=True)
class PlaneTimeSource:
    """One complete per-plane time source in ``(t, c, z)`` order."""

    source: str
    values_tcz: np.ndarray
    unit: str | None
    raw_unit: str | None
    warning_code: str | None


def resolve_plane_time_source(
    ome_scene,
    *,
    filetype: str | None,
    original_metadata: Mapping[str, str] | None = None,
) -> tuple[PlaneTimeSource | None, list[ValidationMessage]]:
    """Choose exactly one complete per-plane timing provenance for a scene."""
    logger.debug("Resolving per-plane time source for scene %s with filetype=%s", ome_scene.name, filetype)
    messages: list[ValidationMessage] = []

    delta_source, delta_messages = _delta_t_source(ome_scene)
    messages.extend(delta_messages)
    if delta_source is not None:
        logger.info("Selected Plane.DeltaT as per-plane time source for scene %s", ome_scene.name)
        return delta_source, messages

    if str(filetype or "").lower() != "vsi":
        logger.debug("No format-specific time source enabled for scene %s", ome_scene.name)
        return None, messages

    vsi_source, vsi_messages = _vsi_original_metadata_source(ome_scene, original_metadata or {})
    messages.extend(vsi_messages)
    if vsi_source is not None:
        logger.info("Selected VSI OriginalMetadata as per-plane time source for scene %s", ome_scene.name)
    else:
        logger.warning("No complete VSI OriginalMetadata time source found for scene %s", ome_scene.name)
    return vsi_source, messages


def scalar_t_from_plane_source(scene_id: str, plane_source: PlaneTimeSource) -> tuple[AxisState | None, list[ValidationMessage]]:
    """Derive a scalar time increment from one full per-plane timing source."""
    logger.debug("Deriving scalar t from %s for scene %s", plane_source.source, scene_id)
    messages: list[ValidationMessage] = []
    values = plane_source.values_tcz
    if values.shape[0] < 2:
        messages.append(
            ValidationMessage(
                level="warning",
                code="t_not_repaired",
                message=f"Scene {scene_id} has fewer than two timepoints; scalar t increment cannot be derived.",
            )
        )
        return None, messages

    per_time = []
    for t_index in range(values.shape[0]):
        block = values[t_index]
        unique = np.unique(np.round(block.astype(np.float64), decimals=12))
        if len(unique) != 1:
            messages.append(
                ValidationMessage(
                    level="warning",
                    code="t_plane_values_vary_within_timepoint",
                    message=(
                        f"Scene {scene_id} {plane_source.source} timing varies across c/z within timepoint {t_index}; "
                        "scalar t repair skipped."
                    ),
                )
            )
            return None, messages
        per_time.append(float(unique[0]))

    diffs = np.diff(np.asarray(per_time, dtype=np.float64))
    if diffs.size == 0 or np.any(~np.isfinite(diffs)):
        messages.append(
            ValidationMessage(
                level="warning",
                code="t_not_repaired",
                message=f"Scene {scene_id} {plane_source.source} timing did not yield finite increments.",
            )
        )
        return None, messages

    median_diff = float(np.median(diffs))
    tolerance = max(1e-9, abs(median_diff) * 1e-6)
    if median_diff <= 0 or np.max(np.abs(diffs - median_diff)) > tolerance:
        messages.append(
            ValidationMessage(
                level="warning",
                code="t_inconsistent_plane_spacing",
                message=(
                    f"Scene {scene_id} {plane_source.source} timing increments are not constant; scalar t repair skipped."
                ),
            )
        )
        return None, messages

    logger.info("Derived scalar t for scene %s from %s: %s %s", scene_id, plane_source.source, median_diff, plane_source.unit)
    return (
        AxisState(
            axis="t",
            value=median_diff,
            unit=plane_source.unit,
            raw_unit=plane_source.raw_unit,
            source=plane_source.source,
            placeholder=False,
            repaired=True,
            confidence="medium",
            warning_code=plane_source.warning_code,
        ),
        messages,
    )


def _delta_t_source(ome_scene) -> tuple[PlaneTimeSource | None, list[ValidationMessage]]:
    logger.debug("Checking Plane.DeltaT completeness for scene %s", ome_scene.name)
    expected = ome_scene.size_t * ome_scene.size_c * ome_scene.size_z
    values = np.full(expected, np.nan, dtype=np.float64)
    raw_units: set[str] = set()
    seen = set()
    for plane in ome_scene.planes:
        index = _plane_index(ome_scene, plane)
        if index is None:
            continue
        seen.add(index)
        raw_value = plane.get("DeltaT")
        if raw_value is None:
            continue
        try:
            values[index] = float(raw_value)
        except Exception:
            return None, [
                ValidationMessage(
                    level="warning",
                    code="plane_delta_t_invalid",
                    message=f"Scene {ome_scene.name} has non-numeric Plane.DeltaT values; plane timing stays unresolved.",
                )
            ]
        raw_unit = plane.get("DeltaTUnit")
        if raw_unit:
            raw_units.add(raw_unit)

    if len(seen) != expected or np.any(~np.isfinite(values)):
        logger.debug("Plane.DeltaT is incomplete for scene %s: expected=%d seen=%d", ome_scene.name, expected, len(seen))
        return None, []
    if len(raw_units) > 1:
        return None, [
            ValidationMessage(
                level="warning",
                code="plane_delta_t_mixed_units",
                message=f"Scene {ome_scene.name} has mixed Plane.DeltaT units: {sorted(raw_units)}.",
            )
        ]

    raw_unit = next(iter(raw_units), None)
    unit, warning_code = normalize_unit(raw_unit)
    if unit in {None, "unknown"}:
        return None, [
            ValidationMessage(
                level="warning",
                code="plane_delta_t_unit_unresolved",
                message=f"Scene {ome_scene.name} Plane.DeltaT units could not be normalized.",
            )
        ]

    logger.debug("Plane.DeltaT is complete for scene %s with unit=%s", ome_scene.name, unit)
    return (
        PlaneTimeSource(
            source="Plane.DeltaT",
            values_tcz=values.reshape((ome_scene.size_t, ome_scene.size_c, ome_scene.size_z)),
            unit=unit,
            raw_unit=raw_unit,
            warning_code=warning_code,
        ),
        [],
    )


def _vsi_original_metadata_source(
    ome_scene,
    original_metadata: Mapping[str, str],
) -> tuple[PlaneTimeSource | None, list[ValidationMessage]]:
    logger.debug("Checking VSI OriginalMetadata timing for scene %s", ome_scene.name)
    expected = ome_scene.size_t * ome_scene.size_c * ome_scene.size_z
    pattern_value = re.compile(rf"^{re.escape(ome_scene.name)} Value #(\d+)$")
    pattern_unit = re.compile(rf"^{re.escape(ome_scene.name)} Units #(\d+)$")
    values_by_id: dict[int, float] = {}
    units_by_id: dict[int, str] = {}

    for key, raw_value in original_metadata.items():
        match = pattern_value.match(key)
        if match is not None:
            index = int(match.group(1))
            if 1 <= index <= expected:
                try:
                    values_by_id[index] = float(raw_value)
                except Exception:
                    return None, [
                        ValidationMessage(
                            level="warning",
                            code="vsi_time_non_numeric",
                            message=f"Scene {ome_scene.name} VSI numbered timing contains non-numeric values.",
                        )
                    ]
            continue
        match = pattern_unit.match(key)
        if match is not None:
            index = int(match.group(1))
            if 1 <= index <= expected:
                units_by_id[index] = raw_value

    if len(values_by_id) != expected:
        logger.debug(
            "VSI timing values are incomplete for scene %s: expected=%d found=%d",
            ome_scene.name,
            expected,
            len(values_by_id),
        )
        return None, [
            ValidationMessage(
                level="warning",
                code="vsi_time_incomplete",
                message=(
                    f"Scene {ome_scene.name} VSI numbered timing expected {expected} values but found {len(values_by_id)}."
                ),
            )
        ]
    if len(units_by_id) != expected:
        logger.debug(
            "VSI timing units are incomplete for scene %s: expected=%d found=%d",
            ome_scene.name,
            expected,
            len(units_by_id),
        )
        return None, [
            ValidationMessage(
                level="warning",
                code="vsi_time_units_incomplete",
                message=(
                    f"Scene {ome_scene.name} VSI numbered timing expected {expected} units but found {len(units_by_id)}."
                ),
            )
        ]

    raw_units = {units_by_id[index] for index in range(1, expected + 1)}
    if len(raw_units) > 1:
        return None, [
            ValidationMessage(
                level="warning",
                code="vsi_time_mixed_units",
                message=f"Scene {ome_scene.name} VSI numbered timing has mixed units: {sorted(raw_units)}.",
            )
        ]

    raw_unit = next(iter(raw_units), None)
    unit, warning_code = normalize_unit(raw_unit)
    if unit in {None, "unknown"}:
        return None, [
            ValidationMessage(
                level="warning",
                code="vsi_time_unit_unresolved",
                message=f"Scene {ome_scene.name} VSI numbered timing units could not be normalized.",
            )
        ]

    values_ctz = np.full((ome_scene.size_c, ome_scene.size_t, ome_scene.size_z), math.nan, dtype=np.float64)
    counter = 1
    for c_index in range(ome_scene.size_c):
        for t_index in range(ome_scene.size_t):
            for z_index in range(ome_scene.size_z):
                values_ctz[c_index, t_index, z_index] = values_by_id[counter]
                counter += 1
    values_tcz = np.transpose(values_ctz, (1, 0, 2))
    logger.debug("VSI OriginalMetadata timing is complete for scene %s with unit=%s", ome_scene.name, unit)
    return (
        PlaneTimeSource(
            source="OriginalMetadata.VSI",
            values_tcz=values_tcz,
            unit=unit,
            raw_unit=raw_unit,
            warning_code=warning_code,
        ),
        [],
    )


def _plane_index(ome_scene, plane: Mapping[str, str | None]) -> int | None:
    try:
        t = int(plane.get("TheT") or 0)
        c = int(plane.get("TheC") or 0)
        z = int(plane.get("TheZ") or 0)
    except Exception:
        return None
    if not (0 <= t < ome_scene.size_t and 0 <= c < ome_scene.size_c and 0 <= z < ome_scene.size_z):
        return None
    return (t * ome_scene.size_c * ome_scene.size_z) + (c * ome_scene.size_z) + z
