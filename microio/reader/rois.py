"""Reader helpers for microio ROI image groups."""

from __future__ import annotations

import logging

from microio.common.models import DatasetHandle, RoiReadResult
from microio.common.ngff import flattened_attrs, ome_metadata


logger = logging.getLogger("microio.reader.rois")


def list_rois(ds: DatasetHandle, scene: int | str) -> list[str]:
    """List ROI image names stored under one scene."""
    ref = ds.scene_ref(scene)
    scene_group = ds.root[ref.id]
    if "rois" not in scene_group:
        logger.debug("Scene %s has no rois group", ref.id)
        return []
    names = sorted(str(name) for name, _ in scene_group["rois"].groups())
    logger.debug("Found %d ROI groups for scene %s: %s", len(names), ref.id, names)
    return names


def read_roi_metadata(ds: DatasetHandle, scene: int | str, name: str) -> dict[str, object]:
    """Read ROI attrs without loading the image payload."""
    logger.debug("Reading ROI metadata for scene=%s roi=%s", scene, name)
    group = _roi_group(ds, scene, name)
    attrs = flattened_attrs(group)
    return {
        "scene_id": ds.scene_ref(scene).id,
        "roi_name": str(name),
        "level_path": "0",
        "shape": tuple(int(dim) for dim in group["0"].shape),
        "attrs": attrs,
        "microio": dict(attrs.get("microio", {})),
        "ome": ome_metadata(group),
    }


def load_roi(ds: DatasetHandle, scene: int | str, name: str) -> RoiReadResult:
    """Load one ROI array together with stored metadata."""
    ref = ds.scene_ref(scene)
    group = _roi_group(ds, ref.id, name)
    metadata = read_roi_metadata(ds, ref.id, name)
    logger.debug("Loading ROI %s for scene %s", name, ref.id)
    return RoiReadResult(
        scene_id=ref.id,
        roi_name=str(name),
        level_path="0",
        shape=metadata["shape"],
        array=group["0"][:],
        attrs=metadata["attrs"],
        microio=metadata["microio"],
        ome=metadata["ome"],
    )


def _roi_group(ds: DatasetHandle, scene: int | str, name: str):
    ref = ds.scene_ref(scene)
    scene_group = ds.root[ref.id]
    if "rois" not in scene_group or str(name) not in scene_group["rois"]:
        logger.warning("Requested missing ROI %s for scene %s", name, ref.id)
        raise KeyError(f"Scene {ref.id} has no ROI named {name!r}; available={list_rois(ds, ref.id)}")
    return scene_group["rois"][str(name)]
