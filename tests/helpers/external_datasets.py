"""Helpers for optional regression tests against user-supplied external datasets."""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def clone_scene_subset(source: str | Path, destination: str | Path, scene_ids: list[str]) -> Path:
    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists():
        shutil.rmtree(destination_path)
    destination_path.mkdir(parents=True, exist_ok=True)

    for name in (".zgroup", ".zattrs", "zarr.json"):
        src = source_path / name
        if src.exists():
            shutil.copy2(src, destination_path / name)

    ome_source = source_path / "OME"
    if ome_source.exists():
        shutil.copytree(ome_source, destination_path / "OME", copy_function=shutil.copy2)
        _rewrite_series_metadata(destination_path / "OME", scene_ids)

    for scene_id in scene_ids:
        _clone_scene_tree(source_path / scene_id, destination_path / scene_id)
    return destination_path


def _rewrite_series_metadata(ome_dir: Path, scene_ids: list[str]) -> None:
    zattrs = ome_dir / ".zattrs"
    if zattrs.exists():
        data = json.loads(zattrs.read_text(encoding="utf-8"))
        data["series"] = [str(scene_id) for scene_id in scene_ids]
        zattrs.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _clone_scene_tree(source: Path, destination: Path) -> None:
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
            shutil.copy2(entry, target)
            continue
        _link_or_copy(entry, target)


def _should_copy_metadata_file(name: str) -> bool:
    return name in {".zattrs", ".zgroup", "zarr.json", ".zarray", ".zmetadata"}


def _should_junction_directory(path: Path) -> bool:
    if path.name == "tables":
        return False
    return path.name.isdigit()


def _junction_or_clone_dir(source: Path, destination: Path) -> None:
    try:
        _create_directory_junction(source, destination)
    except Exception:
        _clone_scene_tree(source, destination)


def _link_or_copy(source: Path, destination: Path) -> None:
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def _create_directory_junction(source: Path, destination: Path) -> None:
    if destination.exists():
        raise FileExistsError(f"Destination already exists: {destination}")
    if sys.platform != "win32":
        raise OSError("Directory junctions are only supported on Windows")
    if source.drive.lower() != destination.drive.lower():
        raise OSError("Directory junctions require source and destination on the same drive")
    destination.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(destination), str(source)],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise OSError(result.stdout.strip() or result.stderr.strip())
