"""Java runtime setup for Bio-Formats based writer functionality."""

from __future__ import annotations

import logging
import os
from pathlib import Path


logger = logging.getLogger("microio.writer.setup")


def setup_java(
    fetch_mode: str = "never",
    java_version: str = "21",
    java_home: str | None = None,
    logback_path: str | None = None,
    jvm_options: list[str] | None = None,
) -> None:
    """Configure the Java runtime before Bio-Formats starts the JVM.

    Parameters
    ----------
    fetch_mode:
        Strategy passed to ``scyjava`` for resolving a JVM installation.
    java_version:
        Requested Java major version constraint.
    java_home:
        Optional explicit Java home directory. When provided, ``JAVA_HOME`` and
        ``PATH`` are updated before ``scyjava`` is configured.
    logback_path:
        Optional path to a Logback XML file used by the Java-side Bio-Formats
        logging stack.
    jvm_options:
        Additional JVM options appended after the mandatory ``--add-opens``
        flags required by Bio-Formats on modern JVMs.
    """
    import scyjava

    logger.info(
        "Configuring Java runtime for Bio-Formats (fetch_mode=%s, java_version=%s, java_home=%s)",
        fetch_mode,
        java_version,
        java_home or "<system>",
    )
    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{Path(java_home) / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"
        logger.debug("Set JAVA_HOME to %s", java_home)

    scyjava.config.set_java_constraints(fetch=fetch_mode, version=java_version)
    logger.debug("Applied scyjava Java constraints")

    if logback_path:
        scyjava.config.add_option(f"-Dlogback.configurationFile={logback_path}")
        logger.info("Configured Logback file: %s", logback_path)

    # Required by Bio-Formats internals on modern JVMs.
    scyjava.config.add_option("--add-opens=java.base/java.util.regex=ALL-UNNAMED")
    scyjava.config.add_option("--add-opens=java.base/java.lang=ALL-UNNAMED")
    scyjava.config.add_option("--add-opens=java.base/java.util=ALL-UNNAMED")

    if jvm_options:
        for option in jvm_options:
            scyjava.config.add_option(option)
            logger.debug("Added JVM option: %s", option)
