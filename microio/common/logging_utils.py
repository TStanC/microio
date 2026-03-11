"""Logging helpers for consistent microio output formatting."""

from __future__ import annotations

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure package logging and return the root ``microio`` logger.

    Parameters
    ----------
    level:
        Textual logging level such as ``"DEBUG"``, ``"INFO"``, ``"WARNING"``,
        or ``"ERROR"``. Unknown values fall back to ``INFO``.

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
    resolved_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=resolved_level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger = logging.getLogger("microio")
    logger.setLevel(resolved_level)
    return logger
