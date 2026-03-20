"""Image writers for NGFF label images and microio ROI cutouts."""

from __future__ import annotations

from copy import deepcopy
import logging
from typing import Any

import dask.array as da
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
    group_zarr_format,
    maybe_cast_array,
    normalize_slice_spec,
    read_node_ome_metadata,
    replace_node_ome_metadata,
    require_child_group,
    require_writeable_scene,
    source_level_metadata,
    source_pyramid_metadata,
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
    colors: list[dict[str, Any]] | None = None,
    properties: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
    threads: int | None = None,
) -> LabelWriteReport:
    """Write an NGFF-compliant label pyramid under ``labels/<name>``.

    Parameters
    ----------
    ds:
        Open dataset handle opened in a writable mode.
    scene:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    name:
        Label image name to create under the scene ``labels/`` group.
    data:
        Integer-valued label image for source level ``0``. The shape must match
        the source level-0 image exactly. The library expects the dataset axis
        order ``("t", "c", "z", "y", "x")`` and therefore expects the label
        data to follow that same order.
    source_level:
        Source multiscale level. Only ``0`` is accepted.
    chunks:
        Optional chunk shape override for the written label arrays.
    dtype:
        Optional integer dtype to cast ``data`` to before writing.
    attrs:
        Optional extra non-OME attributes to store on the label group, for
        example ``{"description": "nucleus segmentation"}``.
    colors:
        Optional NGFF image-label color metadata, for example
        ``[{"label-value": 0, "rgba": [0, 0, 0, 0]}, {"label-value": 7, "rgba": [255, 255, 0, 255]}]``.
    properties:
        Optional NGFF image-label properties metadata, for example
        ``[{"label-value": 7, "class": "nucleus"}]``.
    overwrite:
        Whether to replace an existing label image with the same name.
    threads:
        Optional worker count for array writes.

    Returns
    -------
    LabelWriteReport
        Structured summary of the written label image.

    Notes
    -----
    The function writes the level-0 label image directly and derives coarser
    label levels from the source image pyramid metadata. Existing targets are
    preserved unless ``overwrite=True`` is supplied.
    """
    if str(source_level) != "0":
        raise ValueError("Label pyramids must be derived from source level 0 to remain NGFF-consistent")

    ref = require_writeable_scene(ds, scene)
    logger.info("Writing label image %s for scene %s", name, ref.id)
    _, multiscale, source_datasets = source_pyramid_metadata(ds, ref.id)

    array_like = maybe_cast_array(coerce_array(data), dtype)
    level0 = ds.level_ref(ref.id, 0)
    shape = tuple(int(dim) for dim in array_like.shape)
    if shape != tuple(int(dim) for dim in level0.shape):
        raise ValueError(f"Label image shape {shape} does not match source level shape {level0.shape}")

    dtype_obj = np.dtype(array_like.dtype)
    if not np.issubdtype(dtype_obj, np.integer):
        raise ValueError(f"Label images must use an integer dtype, received {dtype_obj}")

    _validate_label_name(name)
    resolved_colors = _normalize_label_colors(colors)
    resolved_properties = _normalize_label_properties(properties)

    labels = require_child_group(ds.root[ref.id], "labels")
    ensure_group_absent_or_overwrite(labels, name, overwrite=overwrite)
    label_group = labels.create_group(name)

    axes = tuple(axis["name"] for axis in multiscale["axes"])
    base = array_like
    written_paths: list[str] = []
    for dataset_md in source_datasets:
        path = str(dataset_md["path"])
        target_shape = tuple(int(dim) for dim in ds.level_ref(ref.id, path).shape)
        level_data = _resample_label_level(base, target_shape)
        level_dtype = np.dtype(level_data.dtype)
        level_chunks = default_chunks(target_shape, level_dtype, chunks)
        logger.debug(
            "Writing label level %s for %s/%s with shape=%s chunks=%s",
            path,
            ref.id,
            name,
            target_shape,
            level_chunks,
        )
        write_array(
            label_group,
            path,
            level_data,
            chunks=level_chunks,
            threads=threads,
            dimension_names=axes if group_zarr_format(label_group) >= 3 else None,
        )
        written_paths.append(path)

    replace_node_ome_metadata(
        labels,
        {"labels": _updated_label_listing(labels, name)},
    )
    replace_node_ome_metadata(
        label_group,
        _label_group_ome_metadata(
            multiscale=multiscale,
            source_datasets=source_datasets,
            name=name,
            colors=resolved_colors,
            properties=resolved_properties,
        ),
        extra_attrs=_label_extra_attrs(ref.id, level0.path, attrs),
    )
    ds.invalidate_caches(scene_id=ref.id)
    return LabelWriteReport(
        scene_id=ref.id,
        label_name=name,
        level_path=written_paths[0],
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
    """Write a single-scale ROI cutout under ``rois/<name>/0``.

    Parameters
    ----------
    ds:
        Open dataset handle opened in a writable mode.
    scene:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    name:
        ROI name to create under the scene ``rois/`` group.
    slices:
        Mapping from axis name to a slice specification. Supported values are
        ``slice`` objects, ``(start, stop)`` tuples, or integer indices.
        Example: ``{"t": 0, "z": (0, 4), "y": (100, 300), "x": (200, 400)}``.
    source_level:
        Source multiscale level used for the cutout.
    chunks:
        Optional chunk shape override for the written ROI array.
    attrs:
        Optional extra non-OME attributes to store on the ROI group.
    overwrite:
        Whether to replace an existing ROI with the same name.
    threads:
        Optional worker count for array writes.

    Returns
    -------
    RoiWriteReport
        Structured summary of the written ROI cutout.

    Notes
    -----
    Integer axis selectors are normalized to length-1 slices so the ROI keeps
    the original axis count instead of squeezing dimensions away.
    """
    ref = require_writeable_scene(ds, scene)
    logger.info("Writing ROI %s for scene %s from source level %s", name, ref.id, source_level)
    _, multiscale, dataset_md = source_level_metadata(ds, ref.id, source_level)
    axes = [axis["name"] for axis in multiscale["axes"]]
    source = ds.read_level(ref.id, source_level)
    normalized, provenance = _normalize_roi_slices(source.shape, axes, slices)
    roi = coerce_array(source[normalized])
    dtype_obj = np.dtype(roi.dtype)
    resolved_chunks = default_chunks(tuple(int(dim) for dim in roi.shape), dtype_obj, chunks)
    logger.debug("ROI %s for scene %s resolved to shape=%s chunks=%s", name, ref.id, roi.shape, resolved_chunks)

    rois = require_child_group(ds.root[ref.id], "rois")
    ensure_group_absent_or_overwrite(rois, name, overwrite=overwrite)
    group = rois.create_group(name)
    write_array(
        group,
        "0",
        roi,
        chunks=resolved_chunks,
        threads=threads,
        dimension_names=tuple(axes) if group_zarr_format(group) >= 3 else None,
    )

    replace_node_ome_metadata(
        group,
        _roi_group_ome_metadata(multiscale, dataset_md, name),
        extra_attrs=_roi_extra_attrs(ref.id, source_level, axes, provenance, dataset_md, attrs),
    )
    ds.invalidate_caches(scene_id=ref.id)
    return RoiWriteReport(
        scene_id=ref.id,
        roi_name=name,
        level_path="0",
        shape=tuple(int(dim) for dim in roi.shape),
        persisted=True,
    )


def _label_group_ome_metadata(
    *,
    multiscale: dict[str, Any],
    source_datasets: list[dict[str, Any]],
    name: str,
    colors: list[dict[str, Any]] | None,
    properties: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    """Build semantic OME metadata for a label image group.

    Returns a semantic metadata dictionary containing ``multiscales`` and
    ``image-label`` blocks ready to be serialized through the shared NGFF
    helpers.
    """
    label_md = {
        "source": {"image": "../../"},
        "version": _image_label_version(multiscale),
    }
    if colors is not None:
        label_md["colors"] = colors
    if properties is not None:
        label_md["properties"] = properties
    return {
        "multiscales": [_pyramid_multiscale(multiscale, source_datasets, name)],
        "image-label": label_md,
    }


def _roi_group_ome_metadata(multiscale: dict[str, Any], dataset_md: dict[str, Any], name: str) -> dict[str, Any]:
    """Build semantic OME metadata for a single-scale ROI image group."""
    roi_dataset_md = deepcopy(dataset_md)
    roi_dataset_md["path"] = "0"
    return {"multiscales": [_single_scale_multiscale(multiscale, roi_dataset_md, name)]}


def _label_extra_attrs(scene_id: str, source_level_path: str, attrs: dict[str, Any] | None) -> dict[str, Any]:
    microio = {
        "schema": MICROIO_WRITER_LABEL_SCHEMA,
        "schema_version": MICROIO_WRITER_LABEL_SCHEMA_VERSION,
        "source_scene_id": scene_id,
        "source_level": str(source_level_path),
    }
    extra = {"microio": microio}
    if attrs:
        extra.update(attrs)
    return extra


def _roi_extra_attrs(
    scene_id: str,
    source_level: int | str,
    axes: list[str],
    provenance: dict[str, Any],
    dataset_md: dict[str, Any],
    attrs: dict[str, Any] | None,
) -> dict[str, Any]:
    microio = {
        "schema": MICROIO_WRITER_ROI_SCHEMA,
        "schema_version": MICROIO_WRITER_ROI_SCHEMA_VERSION,
        "source_scene_id": scene_id,
        "source_level": str(source_level),
        "source_axes": axes,
        "slices": provenance,
        "origin": _origin_from_provenance(provenance),
        "voxel_size": dataset_md["coordinateTransformations"][0]["scale"],
    }
    extra = {"microio": microio}
    if attrs:
        extra.update(attrs)
    return extra


def _single_scale_multiscale(multiscale: dict[str, Any], dataset_md: dict[str, Any], name: str) -> dict[str, Any]:
    """Build a single-scale multiscales block derived from one source level.

    The output mirrors the source scene's axis metadata while rewriting the
    dataset path to ``"0"`` for the ROI-local image.
    """
    axes = deepcopy(multiscale["axes"])
    dataset = deepcopy(dataset_md)
    dataset["path"] = "0"
    out = {
        "axes": axes,
        "name": name,
        "datasets": [dataset],
    }
    if "metadata" in multiscale:
        out["metadata"] = deepcopy(multiscale["metadata"])
    if "type" in multiscale:
        out["type"] = deepcopy(multiscale["type"])
    if "coordinateTransformations" in multiscale:
        out["coordinateTransformations"] = deepcopy(multiscale["coordinateTransformations"])
    if "version" in multiscale:
        out["version"] = multiscale["version"]
    return out


def _pyramid_multiscale(multiscale: dict[str, Any], source_datasets: list[dict[str, Any]], name: str) -> dict[str, Any]:
    """Build a multiscales block for a derived label pyramid.

    This preserves source-scene metadata fields such as ``metadata`` and
    ``type`` while rewriting the multiscale ``name``.
    """
    datasets = [deepcopy(item) for item in source_datasets]
    out = {
        "axes": deepcopy(multiscale["axes"]),
        "name": name,
        "datasets": datasets,
    }
    if "metadata" in multiscale:
        out["metadata"] = deepcopy(multiscale["metadata"])
    if "type" in multiscale:
        out["type"] = deepcopy(multiscale["type"])
    if "coordinateTransformations" in multiscale:
        out["coordinateTransformations"] = deepcopy(multiscale["coordinateTransformations"])
    if "version" in multiscale:
        out["version"] = multiscale["version"]
    return out


def _updated_label_listing(labels_group, name: str) -> list[str]:
    """Return the sorted label-image listing for a scene ``labels`` container."""
    current = read_node_ome_metadata(labels_group).get("labels", [])
    listed = {str(item) for item in current if str(item)}
    listed.add(name)
    return sorted(listed)


def _normalize_roi_slices(shape: tuple[int, ...], axes: list[str], slices: dict[str, Any]) -> tuple[tuple[Any, ...], dict[str, Any]]:
    """Normalize ROI slice specs and capture provenance in source coordinates.

    The returned provenance dictionary is later stored under the microio ROI
    extension block so consumers can reconstruct the source-space crop.
    """
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


def _resample_label_level(data: Any, target_shape: tuple[int, ...]):
    """Nearest-neighbor downsample a label image to a target multiscale shape.

    Label data is resampled with index-based nearest-neighbor selection to
    avoid introducing fractional or blended label ids.
    """
    current_shape = tuple(int(dim) for dim in data.shape)
    if current_shape == target_shape:
        return data

    result = data
    for axis, (src, dst) in enumerate(zip(current_shape, target_shape, strict=True)):
        if dst <= 0:
            raise ValueError(f"Target label shape contains a non-positive dimension: {target_shape}")
        if dst == src:
            continue
        indices = ((np.arange(dst, dtype=np.int64) * src) // dst).astype(np.int64)
        if isinstance(result, da.Array):
            result = da.take(result, indices, axis=axis)
        else:
            result = np.take(np.asarray(result), indices, axis=axis)
    return result


def _normalize_label_colors(colors: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Validate image-label color metadata.

    Each entry must define ``label-value`` and may optionally define ``rgba``
    as four integers in the inclusive range ``0..255``.
    """
    if colors is None:
        return None
    if not isinstance(colors, list):
        raise TypeError("Label colors must be provided as a list of dictionaries")
    normalized: list[dict[str, Any]] = []
    seen: set[int] = set()
    for item in colors:
        if not isinstance(item, dict):
            raise TypeError("Each label color entry must be a dictionary")
        if "label-value" not in item:
            raise ValueError("Each label color entry must define 'label-value'")
        label_value = _coerce_label_value(item["label-value"], field="label-value")
        if label_value in seen:
            raise ValueError(f"Duplicate label color entry for value {label_value}")
        seen.add(label_value)
        entry = deepcopy(item)
        entry["label-value"] = label_value
        rgba = entry.get("rgba")
        if rgba is not None:
            if not isinstance(rgba, list) or len(rgba) != 4:
                raise ValueError(f"Label color rgba for value {label_value} must be a list of 4 integers")
            validated = []
            for channel in rgba:
                if not isinstance(channel, int) or not (0 <= channel <= 255):
                    raise ValueError(f"RGBA values for label {label_value} must be integers between 0 and 255")
                validated.append(channel)
            entry["rgba"] = validated
        normalized.append(entry)
    return normalized


def _normalize_label_properties(properties: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """Validate image-label property metadata.

    Each entry must define ``label-value`` and may carry any additional
    JSON-serializable label annotation fields.
    """
    if properties is None:
        return None
    if not isinstance(properties, list):
        raise TypeError("Label properties must be provided as a list of dictionaries")
    normalized: list[dict[str, Any]] = []
    for item in properties:
        if not isinstance(item, dict):
            raise TypeError("Each label properties entry must be a dictionary")
        if "label-value" not in item:
            raise ValueError("Each label properties entry must define 'label-value'")
        entry = deepcopy(item)
        entry["label-value"] = _coerce_label_value(item["label-value"], field="label-value")
        normalized.append(entry)
    return normalized


def _coerce_label_value(value: Any, *, field: str) -> int:
    """Coerce a label metadata value to a plain integer."""
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"Label metadata field {field!r} must be an integer")
    return int(value)


def _image_label_version(multiscale: dict[str, Any]) -> str:
    """Choose the image-label schema version matching the source metadata dialect."""
    multiscale_version = str(multiscale.get("version", "0.5"))
    return "0.4" if multiscale_version.startswith("0.4") else "0.5"


def _validate_label_name(name: str) -> None:
    """Reject empty label names."""
    if not str(name):
        raise ValueError("Label names must be non-empty")
