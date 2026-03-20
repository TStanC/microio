"""Unit normalization helpers with deterministic coercion behavior."""

from __future__ import annotations

import re

from .constants import UNIT_UNKNOWN

_CANONICAL_UNIT_MAP = {
    "micrometer": "micrometer",
    "micrometre": "micrometer",
    "um": "micrometer",
    "µm": "micrometer",
    "âµm": "micrometer",
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
    "ç— ": "micrometer",
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
    """Normalize a raw unit token and return ``(normalized, warning_code)``.

    Parameters
    ----------
    raw:
        Raw unit token from OME-XML or Zarr metadata, for example ``"um"``,
        ``"micrometer"``, ``"s"``, or a known corrupted token such as
        ``"ç— "``.

    Returns
    -------
    tuple[str | None, str | None]
        A pair of ``(normalized_unit, warning_code)``. Examples:

        - ``("micrometer", None)`` for canonical tokens
        - ``("micrometer", "unit_corrupted_coerced")`` for known corrupted
          tokens
        - ``("unknown", "unit_unknown")`` for unsupported tokens
        - ``(None, "unit_missing")`` when no token is present
    """
    if raw is None:
        return None, "unit_missing"
    token = raw.strip()
    if not token:
        return None, "unit_empty"

    lowered = token.lower()
    if lowered in _CANONICAL_UNIT_MAP:
        return _CANONICAL_UNIT_MAP[lowered], None

    if token in _CORRUPTED_UNIT_HINTS:
        return _CORRUPTED_UNIT_HINTS[token], "unit_corrupted_coerced"

    cleaned = re.sub(r"\s+", "", lowered)
    if cleaned in _CANONICAL_UNIT_MAP:
        return _CANONICAL_UNIT_MAP[cleaned], "unit_cleaned"

    return UNIT_UNKNOWN, "unit_unknown"


def to_base_scale(value: float, normalized_unit: str | None) -> float:
    """Convert value to base scale for its unit family.

    Spatial values are converted to micrometers and temporal values to
    seconds. Unknown units are returned unchanged.
    """
    if normalized_unit is None or normalized_unit == UNIT_UNKNOWN:
        return value
    return value * _SCALE_TO_BASE.get(normalized_unit, 1.0)
