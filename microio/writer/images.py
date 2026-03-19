"""Image writers for labels and ROIs."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import numpy as np

from microio.common.constants import (
    MICROIO_WRITER_LABEL_SCHEMA,
    MICROIO_WRITER_LABEL_SCHEMA_VERSION,
    MICROIO_WRITER_ROI_SCHEMA,
    MICROIO_WRITER_ROI_SCHEMA_VERSION,
)
from microio.common.models import LabelWriteReport, RoiWriteReport
from microio.writer.common import (
    coerce_array,
    default_chunks,
    ensure_group_absent_or_overwrite,
    maybe_cast_array,
    normalize_slice_spec,
    require_child_group,
    require_writeable_scene,
    source_level_metadata,
    write_array,
)


logger = logging.getLogger("microio.writer.images")


def write_label_image(
    ds,
    scene: int | str,
    name: str,
    data: Any,
    *,
    source_level: int | str = 0,
    chunks: tuple[int, ...] | None = None,
    dtype: Any | None = None,
    attrs: dict[str, Any] | None = None,
    overwrite: bool = False,
    threads: int | None = None,
) -> LabelWriteReport:
    """Write a full-resolution derived image under ``labels/<name>/0``."""
    ref = require_writeable_scene(ds, scene)
    logger.info("Writing label image %s for scene %s", name, ref.id)
    _, multiscale, dataset_md = source_level_metadata(ds, ref.id, source_level)
    array_like = maybe_cast_array(coerce_array(data), dtype)
    shape = tuple(int(dim) for dim in array_like.shape)
    source_shape = tuple(int(dim) for dim in ds.level_ref(ref.id, source_level).shape)
    if shape != source_shape:
        raise ValueError(f"Label image shape {shape} does not match source level shape {source_shape}")

    dtype_obj = np.dtype(array_like.dtype)
    resolved_chunks = default_chunks(shape, dtype_obj, chunks)
    logger.debug("Label image %s for scene %s uses chunks=%s", name, ref.id, resolved_chunks)
    labels = require_child_group(ds.root[ref.id], "labels")
    ensure_group_absent_or_overwrite(labels, name, overwrite=overwrite)
    group = labels.create_group(name)
    write_array(group, "0", array_like, chunks=resolved_chunks, threads=threads)

    group.attrs.update(_single_scale_metadata(multiscale, dataset_md, name))
    group.attrs["microio"] = {
        "schema": MICROIO_WRITER_LABEL_SCHEMA,
        "schema_version": MICROIO_WRITER_LABEL_SCHEMA_VERSION,
        "source_scene_id": ref.id,
        "source_level": str(ds.level_ref(ref.id, source_level).path),
    }
    if attrs:
        group.attrs.update(attrs)
    ds.invalidate_caches(scene_id=ref.id)
    return LabelWriteReport(
        scene_id=ref.id,
        label_name=name,
        level_path="0",
        shape=shape,
        dtype=str(dtype_obj),
        persisted=True,
    )


def write_roi(
    ds,
    scene: int | str,
    name: str,
    slices: dict[str, Any],
    *,
    source_level: int | str = 0,
    chunks: tuple[int, ...] | None = None,
    attrs: dict[str, Any] | None = None,
    overwrite: bool = False,
    threads: int | None = None,
) -> RoiWriteReport:
    """Write a single-scale ROI cutout under ``rois/<name>/0``."""
    ref = require_writeable_scene(ds, scene)
    logger.info("Writing ROI %s for scene %s from source level %s", name, ref.id, source_level)
    _, multiscale, dataset_md = source_level_metadata(ds, ref.id, source_level)
    axes = [axis["name"] for axis in multiscale["axes"]]
    source = ds.read_level(ref.id, source_level)
    normalized, provenance = _normalize_roi_slices(source.shape, axes, slices)
    roi = source[normalized]
    roi = coerce_array(roi)
    dtype_obj = np.dtype(roi.dtype)
    resolved_chunks = default_chunks(tuple(int(dim) for dim in roi.shape), dtype_obj, chunks)
    logger.debug("ROI %s for scene %s resolved to shape=%s chunks=%s", name, ref.id, roi.shape, resolved_chunks)

    rois = require_child_group(ds.root[ref.id], "rois")
    ensure_group_absent_or_overwrite(rois, name, overwrite=overwrite)
    group = rois.create_group(name)
    write_array(group, "0", roi, chunks=resolved_chunks, threads=threads)

    roi_dataset_md = deepcopy(dataset_md)
    roi_dataset_md["path"] = "0"
    group.attrs.update(_single_scale_metadata(multiscale, roi_dataset_md, name))
    group.attrs["microio"] = {
        "schema": MICROIO_WRITER_ROI_SCHEMA,
        "schema_version": MICROIO_WRITER_ROI_SCHEMA_VERSION,
        "source_scene_id": ref.id,
        "source_level": str(ds.level_ref(ref.id, source_level).path),
        "source_axes": axes,
        "slices": provenance,
        "origin": _origin_from_provenance(provenance),
        "voxel_size": dataset_md["coordinateTransformations"][0]["scale"],
    }
    if attrs:
        group.attrs.update(attrs)
    ds.invalidate_caches(scene_id=ref.id)
    return RoiWriteReport(
        scene_id=ref.id,
        roi_name=name,
        level_path="0",
        shape=tuple(int(dim) for dim in roi.shape),
        persisted=True,
    )


def _single_scale_metadata(multiscale: dict, dataset_md: dict, name: str) -> dict:
    """Build a single-scale multiscales block derived from a source image level."""
    axes = deepcopy(multiscale["axes"])
    dataset = deepcopy(dataset_md)
    dataset["path"] = "0"
    return {
        "multiscales": [
            {
                "metadata": deepcopy(multiscale.get("metadata", {})),
                "axes": axes,
                "name": name,
                "datasets": [dataset],
                "version": multiscale.get("version", "0.4"),
            }
        ]
    }


def _normalize_roi_slices(shape: tuple[int, ...], axes: list[str], slices: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize ROI slice specs and capture provenance in source coordinates."""
    normalized: list[Any] = []
    provenance: dict[str, Any] = {}
    for axis, size in zip(axes, shape, strict=True):
        spec = normalize_slice_spec(slices.get(axis, slice(None)))
        if isinstance(spec, int):
            if spec < 0 or spec >= size:
                raise ValueError(f"Axis {axis!r} index {spec} is out of bounds for size {size}")
            normalized.append(slice(spec, spec + 1))
            provenance[axis] = {"start": spec, "stop": spec + 1, "step": 1, "indexed": True}
            continue
        start = 0 if spec.start is None else spec.start
        stop = size if spec.stop is None else spec.stop
        step = 1 if spec.step is None else spec.step
        if step <= 0:
            raise ValueError(f"Axis {axis!r} step must be positive")
        if start < 0 or stop > size or start > stop:
            raise ValueError(f"Axis {axis!r} slice {spec!r} is out of bounds for size {size}")
        normalized.append(slice(start, stop, step))
        provenance[axis] = {"start": start, "stop": stop, "step": step, "indexed": False}
    return tuple(normalized), provenance


def _origin_from_provenance(provenance: dict[str, Any]) -> dict[str, int]:
    """Extract per-axis start coordinates from normalized ROI provenance."""
    return {axis: int(spec["start"]) for axis, spec in provenance.items()}
