"""Logging helpers for consistent microio output formatting."""

from __future__ import annotations

import logging


_LEVEL_ALIASES = {
    "WARN": "WARNING",
    "FATAL": "CRITICAL",
}


def _resolve_logging_level(level: str | int) -> int:
    """Normalize user-facing log-level inputs to a logging module constant."""
    if isinstance(level, int):
        return level
    if isinstance(level, str):
        normalized = _LEVEL_ALIASES.get(level.strip().upper(), level.strip().upper())
        return getattr(logging, normalized, logging.INFO)
    return logging.INFO


def setup_logging(level: str | int = "INFO") -> logging.Logger:
    """Configure package logging and return the root ``microio`` logger.

    Parameters
    ----------
    level:
        Logging level selector. Supported values include canonical names such
        as ``"DEBUG"``, ``"INFO"``, ``"WARNING"``, ``"ERROR"``, and
        ``"CRITICAL"``, common aliases such as ``"WARN"`` and ``"FATAL"``,
        and integer logging levels accepted by :mod:`logging`. Unknown values
        fall back to ``INFO``.

    Returns
    -------
    logging.Logger
        The package logger named ``microio``. Child loggers inherit the
        configured level and handlers.

    Notes
    -----
    ``logging.basicConfig`` only affects the first call in a process. This
    helper still updates the package logger level each time so callers can
    adjust verbosity across invocations.
    """
    resolved_level = _resolve_logging_level(level)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("microio")
    logger.setLevel(resolved_level)
    return logger
