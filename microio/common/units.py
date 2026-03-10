"""Unit normalization helpers with deterministic coercion behavior."""

from __future__ import annotations

import re

from .constants import UNIT_UNKNOWN

_CANONICAL_UNIT_MAP = {
    "micrometer": "micrometer",
    "micrometre": "micrometer",
    "um": "micrometer",
    "µm": "micrometer",
    "nm": "nanometer",
    "nanometer": "nanometer",
    "s": "second",
    "sec": "second",
    "second": "second",
    "ms": "millisecond",
    "millisecond": "millisecond",
    "min": "minute",
    "minute": "minute",
    "h": "hour",
    "hour": "hour",
}

_CORRUPTED_UNIT_HINTS = {
    "痠": "micrometer",
    "10^-3s^1": "millisecond",
}

_SCALE_TO_BASE = {
    "micrometer": 1.0,
    "nanometer": 1e-3,
    "second": 1.0,
    "millisecond": 1e-3,
    "minute": 60.0,
    "hour": 3600.0,
}


def normalize_unit(raw: str | None) -> tuple[str | None, str | None]:
    """Normalize a raw unit token and return ``(normalized, warning_code)``."""
    if raw is None:
        return None, "unit_missing"
    token = raw.strip().lower()
    if not token:
        return None, "unit_empty"

    if token in _CANONICAL_UNIT_MAP:
        return _CANONICAL_UNIT_MAP[token], None

    if raw in _CORRUPTED_UNIT_HINTS:
        return _CORRUPTED_UNIT_HINTS[raw], "unit_corrupted_coerced"

    cleaned = re.sub(r"\s+", "", token)
    if cleaned in _CANONICAL_UNIT_MAP:
        return _CANONICAL_UNIT_MAP[cleaned], "unit_cleaned"

    return UNIT_UNKNOWN, "unit_unknown"


def to_base_scale(value: float, normalized_unit: str | None) -> float:
    """Convert value to base scale for its unit family."""
    if normalized_unit is None or normalized_unit == UNIT_UNKNOWN:
        return value
    factor = _SCALE_TO_BASE.get(normalized_unit, 1.0)
    return value * factor
