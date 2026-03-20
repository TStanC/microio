"""Helpers for creating scratch subset fixtures used by tests."""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys


logger = logging.getLogger("microio.reader.fixtures")


def clone_scene_subset(source: str | Path, destination: str | Path, scene_ids: list[str]) -> Path:
    """Copy a subset of scenes into a new scratch dataset without touching the source.

    Mutable metadata files are copied, while immutable image payload
    directories may be shared through hardlinks or Windows directory junctions
    to keep fixture setup fast.
    """
    source_path = Path(source)
    destination_path = Path(destination)
    logger.debug(
        "Cloning scene subset from %s to %s for scenes=%s",
        source_path,
        destination_path,
        scene_ids,
    )

    if destination_path.exists():
        logger.debug("Removing existing destination fixture at %s", destination_path)
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    for name in (".zgroup", ".zattrs", "zarr.json"):
        src = source_path / name
        if src.exists():
            logger.debug("Copying dataset root metadata file %s", src)
            shutil.copy2(src, destination_path / name)

    ome_source = source_path / "OME"
    if ome_source.exists():
        logger.debug("Copying OME sidecar directory from %s", ome_source)
        shutil.copytree(ome_source, destination_path / "OME", copy_function=shutil.copy2)
        _rewrite_series_metadata(destination_path / "OME", scene_ids)

    for scene_id in scene_ids:
        scene_source = source_path / scene_id
        logger.debug("Copying scene %s from %s", scene_id, scene_source)
        _clone_scene_tree(scene_source, destination_path / scene_id)

    logger.debug("Finished cloning subset fixture to %s", destination_path)
    return destination_path


def _rewrite_series_metadata(ome_dir: Path, scene_ids: list[str]) -> None:
    zattrs = ome_dir / ".zattrs"
    if zattrs.exists():
        logger.debug("Rewriting OME series metadata in %s for scenes=%s", zattrs, scene_ids)
        data = json.loads(zattrs.read_text(encoding="utf-8"))
        data["series"] = [str(scene_id) for scene_id in scene_ids]
        zattrs.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clone_scene_tree(source: Path, destination: Path) -> None:
    """Clone one selected scene quickly without sharing mutable metadata files."""
    destination.mkdir(parents=True, exist_ok=True)
    for entry in source.iterdir():
        target = destination / entry.name
        if entry.is_dir():
            if _should_junction_directory(entry):
                _junction_or_clone_dir(entry, target)
                continue
            _clone_scene_tree(entry, target)
            continue
        if _should_copy_metadata_file(entry.name):
            logger.debug("Copying mutable metadata file %s", entry)
            shutil.copy2(entry, target)
            continue
        _link_or_copy(entry, target)


def _should_copy_metadata_file(name: str) -> bool:
    return name in {".zattrs", ".zgroup", "zarr.json", ".zarray", ".zmetadata"}


def _should_junction_directory(path: Path) -> bool:
    """Return whether a directory can be safely shared as immutable payload.

    Scene-local ``tables`` data is always cloned so tests can mutate it safely.
    """
    name = path.name
    if name == "tables":
        return False
    return name.isdigit()


def _junction_or_clone_dir(source: Path, destination: Path) -> None:
    """Create a directory junction for immutable payload, falling back to cloning."""
    try:
        _create_directory_junction(source, destination)
        logger.debug("Junctioned %s -> %s", destination, source)
    except Exception:
        logger.debug("Falling back to recursive clone for directory %s", source, exc_info=True)
        _clone_scene_tree(source, destination)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
        logger.debug("Hardlinked %s -> %s", source, destination)
    except OSError:
        logger.debug("Falling back to copy for %s", source)
        shutil.copy2(source, destination)


def _create_directory_junction(source: Path, destination: Path) -> None:
    """Create a Windows directory junction for an immutable subtree.

    This helper is test-only and intentionally Windows-specific because the
    sandbox fixture workflow relies on NTFS directory junctions.
    """
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if sys.platform != "win32":
        raise OSError("Directory junctions are only supported on Windows")
    if source.drive.lower() != destination.drive.lower():
        raise OSError("Directory junctions require source and destination on the same drive")

    destination.parent.mkdir(parents=True, exist_ok=True)
    cmd = ["cmd", "/c", "mklink", "/J", str(destination), str(source)]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        raise OSError(
            f"Failed to create junction {destination} -> {source}: "
            f"{result.stdout.strip()} {result.stderr.strip()}".strip()
        )
