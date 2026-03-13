"""Writer API exports for conversion workflows."""

from __future__ import annotations


__all__ = ["setup_java", "convert_file", "convert_many"]


def __getattr__(name: str):
    if name in {"convert_file", "convert_many"}:
        from .convert import convert_file, convert_many

        return {"convert_file": convert_file, "convert_many": convert_many}[name]
    if name == "setup_java":
        from .setup import setup_java

        return setup_java
    raise AttributeError(name)
