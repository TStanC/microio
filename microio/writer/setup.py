"""Java runtime setup for Bio-Formats based writer functionality."""

from __future__ import annotations

import os
from pathlib import Path


def setup_java(
    fetch_mode: str = "never",
    java_version: str = "21",
    java_home: str | None = None,
    logback_path: str | None = None,
    jvm_options: list[str] | None = None,
) -> None:
    """Configure scyjava before any Bio-Formats JVM startup occurs."""
    import scyjava

    if java_home:
        os.environ["JAVA_HOME"] = java_home
        os.environ["PATH"] = f"{Path(java_home) / 'bin'}{os.pathsep}{os.environ.get('PATH', '')}"

    scyjava.config.set_java_constraints(fetch=fetch_mode, version=java_version)

    if logback_path:
        scyjava.config.add_option(f"-Dlogback.configurationFile={logback_path}")

    # Required by Bio-Formats internals on modern JVMs.
    scyjava.config.add_option("--add-opens=java.base/java.util.regex=ALL-UNNAMED")
    scyjava.config.add_option("--add-opens=java.base/java.lang=ALL-UNNAMED")
    scyjava.config.add_option("--add-opens=java.base/java.util=ALL-UNNAMED")

    if jvm_options:
        for option in jvm_options:
            scyjava.config.add_option(option)
