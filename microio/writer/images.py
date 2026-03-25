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
    validate_write_target_name,
    write_array,
    write_array_region,
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
        Integer-valued label image for source level ``0``. The library expects
        dataset axis order ``("t", "c", "z", "y", "x")`` and therefore expects
        the label data to follow that same order. The label channel axis may
        either match the source image channel size or use a singleton size of
        ``1`` when one label volume applies to all source channels.
    source_level:
        Source multiscale level. Only ``0`` is accepted.
    chunks:
        Optional chunk shape override for the written label arrays.
    dtype:
        Optional integer dtype to cast ``data`` to before writing.
    attrs:
        Optional extra label metadata to store under
        ``label_group.attrs["microio"]["label-attrs"]``, for example
        ``{"description": "nucleus segmentation"}``.
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
    preserved unless ``overwrite=True`` is supplied. This is the whole-image
    write path; for caller-coordinated parallel writes by timepoint, use
    :func:`write_label_timepoint`.
    """
    _validate_label_source_level(source_level)
    ref = require_writeable_scene(ds, scene)
    name = validate_write_target_name(name, kind="Label")
    logger.info("Writing label image %s for scene %s", name, ref.id)
    array_like = maybe_cast_array(coerce_array(data), dtype)
    dtype_obj = _validate_label_array_dtype(array_like)
    shape = tuple(int(dim) for dim in array_like.shape)
    _validate_full_label_shape(shape, tuple(int(dim) for dim in ds.level_ref(ref.id, 0).shape))

    state = _prepare_label_group(
        ds,
        ref.id,
        name,
        label_shape=shape,
        dtype_obj=dtype_obj,
        dtype_requested=dtype is not None,
        chunks=chunks,
        attrs=attrs,
        colors=colors,
        properties=properties,
        overwrite=overwrite,
        write_mode="full",
    )

    written_paths: list[str] = []
    for dataset_md in state["source_datasets"]:
        path = str(dataset_md["path"])
        target_shape = tuple(int(dim) for dim in state["level_shapes"][path])
        level_data = _resample_label_level(array_like, target_shape)
        logger.debug(
            "Writing label level %s for %s/%s with shape=%s",
            path,
            ref.id,
            name,
            target_shape,
        )
        write_array_region(
            state["label_group"][path],
            level_data,
            region=_full_region(target_shape),
            threads=threads,
        )
        written_paths.append(path)

    _persist_written_timepoints(state["label_group"], range(shape[0]))
    ds.invalidate_caches(scene_id=ref.id)
    return LabelWriteReport(
        scene_id=ref.id,
        label_name=name,
        level_path=written_paths[0],
        shape=shape,
        dtype=str(dtype_obj),
        persisted=True,
        initialized=bool(state["initialized"]),
    )


def write_label_timepoint(
    ds,
    scene: int | str,
    name: str,
    data: Any,
    *,
    timepoint: int,
    source_level: int | str = 0,
    chunks: tuple[int, ...] | None = None,
    dtype: Any | None = None,
    attrs: dict[str, Any] | None = None,
    colors: list[dict[str, Any]] | None = None,
    properties: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
    overwrite_timepoint: bool = False,
    threads: int | None = None,
) -> LabelWriteReport:
    """Write one timepoint of an NGFF-compliant label pyramid.

    Parameters
    ----------
    ds:
        Open dataset handle opened in a writable mode.
    scene:
        Scene selector accepted by :meth:`DatasetHandle.scene_ref`.
    name:
        Label image name stored under the scene ``labels/`` group.
    data:
        Integer-valued label image in dataset axis order with a singleton
        leading time dimension. The non-time axes must match source level ``0``
        except that the label channel axis may be ``1`` or the source channel
        size.
    timepoint:
        Source-scene time index to populate.
    source_level:
        Source multiscale level. Only ``0`` is accepted.
    chunks:
        Optional chunk shape override used only when a new label image group is
        initialized in this call.
    dtype:
        Optional integer dtype to cast ``data`` to before writing. Existing
        label images reject dtype changes unless they are recreated through
        ``overwrite=True``.
    attrs:
        Optional extra label metadata to store under
        ``label_group.attrs["microio"]["label-attrs"]`` when the label image is
        initialized in this call.
    colors:
        Optional NGFF image-label color metadata applied when the label image is
        initialized in this call.
    properties:
        Optional NGFF image-label properties metadata applied when the label
        image is initialized in this call.
    overwrite:
        Whether to replace an existing label image group before writing the
        requested timepoint.
    overwrite_timepoint:
        Whether to allow rewriting a timepoint already recorded in the label
        group's ``microio.written_timepoints`` metadata.
    threads:
        Optional worker count for array writes.

    Returns
    -------
    LabelWriteReport
        Structured summary of the written label image update, including the
        written timepoint and whether initialization happened in this call.

    Notes
    -----
    This API is intended for caller-coordinated parallelism across disjoint
    timepoints. The library does not provide cross-process locking or work
    claiming for independent writers targeting the same label group.
    """
    _validate_label_source_level(source_level)
    ref = require_writeable_scene(ds, scene)
    name = validate_write_target_name(name, kind="Label")
    logger.info("Writing label timepoint %s for %s/%s", timepoint, ref.id, name)
    array_like = maybe_cast_array(coerce_array(data), dtype)
    dtype_obj = _validate_label_array_dtype(array_like)
    shape = tuple(int(dim) for dim in array_like.shape)
    if shape[0] != 1:
        raise ValueError(f"Timepoint label writes require a singleton t axis, received shape {shape}")

    state = _prepare_label_group(
        ds,
        ref.id,
        name,
        label_shape=_validate_timepoint_label_shape(
            shape,
            tuple(int(dim) for dim in ds.level_ref(ref.id, 0).shape),
            timepoint=timepoint,
        )[0],
        dtype_obj=dtype_obj,
        dtype_requested=dtype is not None,
        chunks=chunks,
        attrs=attrs,
        colors=colors,
        properties=properties,
        overwrite=overwrite,
        write_mode="timepoint",
    )
    _validate_existing_timepoint_write(
        state["label_group"],
        timepoint=timepoint,
        overwrite_timepoint=overwrite_timepoint,
        initialized=bool(state["initialized"]),
    )

    written_paths: list[str] = []
    for dataset_md in state["source_datasets"]:
        path = str(dataset_md["path"])
        target_shape = tuple(int(dim) for dim in state["level_shapes"][path])
        level_data = _resample_label_level(array_like, (1, *target_shape[1:]))
        region = _timepoint_region(target_shape, timepoint)
        logger.debug("Writing label timepoint %s level %s for %s/%s", timepoint, path, ref.id, name)
        write_array_region(
            state["label_group"][path],
            level_data,
            region=region,
            threads=threads,
        )
        written_paths.append(path)

    _persist_written_timepoints(state["label_group"], [timepoint])
    ds.invalidate_caches(scene_id=ref.id)
    return LabelWriteReport(
        scene_id=ref.id,
        label_name=name,
        level_path=written_paths[0],
        shape=shape,
        dtype=str(dtype_obj),
        persisted=True,
        written_timepoint=int(timepoint),
        initialized=bool(state["initialized"]),
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
    name = validate_write_target_name(name, kind="ROI")
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


def _validate_label_source_level(source_level: int | str) -> None:
    """Reject label writes derived from non-zero source levels."""
    if str(source_level) != "0":
        raise ValueError("Label pyramids must be derived from source level 0 to remain NGFF-consistent")


def _validate_label_array_dtype(data: Any) -> np.dtype:
    """Return the validated integer dtype for one label payload."""
    dtype_obj = np.dtype(data.dtype)
    if not np.issubdtype(dtype_obj, np.integer):
        raise ValueError(f"Label images must use an integer dtype, received {dtype_obj}")
    return dtype_obj


def _validate_full_label_shape(shape: tuple[int, ...], source_shape: tuple[int, ...]) -> str:
    """Validate whole-image label shape against source level ``0``."""
    if len(shape) != len(source_shape):
        raise ValueError(f"Label image shape {shape} does not match source level shape {source_shape}")
    if shape[0] != source_shape[0] or shape[2:] != source_shape[2:]:
        raise ValueError(f"Label image shape {shape} does not match source level shape {source_shape}")
    return _channel_mode(shape[1], source_shape[1])


def _validate_timepoint_label_shape(
    shape: tuple[int, ...],
    source_shape: tuple[int, ...],
    *,
    timepoint: int,
) -> tuple[tuple[int, ...], str]:
    """Validate a singleton-time label payload and expand it to full label shape."""
    if not 0 <= int(timepoint) < int(source_shape[0]):
        raise ValueError(f"Timepoint {timepoint} is out of bounds for source size {source_shape[0]}")
    if len(shape) != len(source_shape):
        raise ValueError(f"Timepoint label image shape {shape} does not match source level rank {source_shape}")
    if shape[0] != 1 or shape[2:] != source_shape[2:]:
        raise ValueError(f"Timepoint label image shape {shape} does not match source level shape {source_shape}")
    channel_mode = _channel_mode(shape[1], source_shape[1])
    return (int(source_shape[0]), int(shape[1]), *[int(dim) for dim in source_shape[2:]]), channel_mode


def _channel_mode(label_c: int, source_c: int) -> str:
    """Classify supported label channel layouts relative to the source image."""
    if label_c == int(source_c):
        return "source"
    if label_c == 1:
        return "singleton"
    raise ValueError(f"Label channel axis must be 1 or match source channel size {source_c}, received {label_c}")


def _prepare_label_group(
    ds,
    scene_id: str,
    name: str,
    *,
    label_shape: tuple[int, ...],
    dtype_obj: np.dtype,
    dtype_requested: bool,
    chunks: tuple[int, ...] | None,
    attrs: dict[str, Any] | None,
    colors: list[dict[str, Any]] | None,
    properties: list[dict[str, Any]] | None,
    overwrite: bool,
    write_mode: str,
):
    """Return an initialized label group plus the derived source-level shapes."""
    _, multiscale, source_datasets = source_pyramid_metadata(ds, scene_id)
    level0 = ds.level_ref(scene_id, 0)
    source_shape = tuple(int(dim) for dim in level0.shape)
    channel_mode = _channel_mode(int(label_shape[1]), int(source_shape[1]))
    resolved_colors = _normalize_label_colors(colors)
    resolved_properties = _normalize_label_properties(properties)

    labels = require_child_group(ds.root[scene_id], "labels")
    initialized = False
    label_group = labels[name] if name in labels else None
    if label_group is not None and overwrite:
        ensure_group_absent_or_overwrite(labels, name, overwrite=True)
        label_group = None

    if label_group is None:
        label_group = labels.create_group(name)
        level_shapes = _label_level_shapes(ds, scene_id, label_shape, source_datasets)
        axes = tuple(axis["name"] for axis in multiscale["axes"])
        for dataset_md in source_datasets:
            path = str(dataset_md["path"])
            target_shape = level_shapes[path]
            target_chunks = _label_chunks_from_source(
                ds,
                scene_id,
                path,
                target_shape,
                chunks,
            )
            label_group.create_array(
                path,
                shape=target_shape,
                dtype=dtype_obj,
                chunks=target_chunks,
                dimension_names=axes if group_zarr_format(label_group) >= 3 else None,
                overwrite=True,
                write_data=False,
            )
        if write_mode == "timepoint":
            level0_chunks = tuple(int(chunk) for chunk in label_group["0"].chunks)
            if level0_chunks and level0_chunks[0] > 1:
                logger.warning(
                    "Label %s for scene %s was initialized with time chunks spanning %d frames; "
                    "timepoint writes remain valid but may incur extra write amplification.",
                    name,
                    scene_id,
                    level0_chunks[0],
                )
        replace_node_ome_metadata(labels, {"labels": _updated_label_listing(labels, name)})
        replace_node_ome_metadata(
            label_group,
            _label_group_ome_metadata(
                multiscale=multiscale,
                source_datasets=source_datasets,
                name=name,
                colors=resolved_colors,
                properties=resolved_properties,
            ),
            extra_attrs=_label_extra_attrs(
                scene_id,
                level0.path,
                attrs,
                channel_mode=channel_mode,
                write_mode=write_mode,
                written_timepoints=[],
            ),
        )
        initialized = True
    else:
        if write_mode != "timepoint":
            raise FileExistsError(f"{label_group.path} already exists; pass overwrite=True to replace it.")
        if dtype_requested or chunks is not None or attrs is not None or colors is not None or properties is not None:
            raise ValueError(
                "Existing label images cannot accept dtype, chunks, attrs, colors, or properties updates unless overwrite=True"
            )
        level_shapes = _label_level_shapes(ds, scene_id, label_shape, source_datasets)
        _validate_existing_label_group(label_group, level_shapes, dtype_obj)

    return {
        "label_group": label_group,
        "source_datasets": source_datasets,
        "level_shapes": _label_level_shapes(ds, scene_id, label_shape, source_datasets),
        "initialized": initialized,
    }


def _validate_existing_label_group(label_group, level_shapes: dict[str, tuple[int, ...]], dtype_obj: np.dtype) -> None:
    """Ensure an existing label group matches the expected pyramid layout."""
    microio = label_group.attrs.asdict().get("microio", {})
    if "written_timepoints" not in microio:
        raise ValueError("Existing label image lacks microio timepoint tracking; pass overwrite=True to recreate it.")
    for path, shape in level_shapes.items():
        if path not in label_group:
            raise ValueError(f"Existing label image is missing pyramid level {path!r}")
        array = label_group[path]
        if tuple(int(dim) for dim in array.shape) != shape:
            raise ValueError(f"Existing label level {path!r} shape {array.shape} does not match expected {shape}")
        if np.dtype(array.dtype) != dtype_obj:
            raise ValueError(f"Existing label dtype {array.dtype} does not match incoming dtype {dtype_obj}")


def _label_level_shapes(ds, scene_id: str, label_shape: tuple[int, ...], source_datasets: list[dict[str, Any]]) -> dict[str, tuple[int, ...]]:
    """Return expected label-array shapes for every pyramid level."""
    out: dict[str, tuple[int, ...]] = {}
    for dataset_md in source_datasets:
        path = str(dataset_md["path"])
        source_shape = list(int(dim) for dim in ds.level_ref(scene_id, path).shape)
        source_shape[1] = int(label_shape[1])
        out[path] = tuple(source_shape)
    return out


def _label_chunks_from_source(
    ds,
    scene_id: str,
    level_path: str,
    target_shape: tuple[int, ...],
    chunks: tuple[int, ...] | None,
) -> tuple[int, ...]:
    """Choose label-array chunks by inheriting the source image chunk layout."""
    if chunks is not None:
        return tuple(int(max(1, min(dim, chunk))) for dim, chunk in zip(target_shape, chunks, strict=True))
    source_chunks = tuple(int(chunk) for chunk in ds.read_level_zarr(scene_id, level_path).chunks)
    return tuple(min(int(dim), int(chunk)) for dim, chunk in zip(target_shape, source_chunks, strict=True))


def _full_region(shape: tuple[int, ...]) -> tuple[slice, ...]:
    """Return a region covering a full array shape."""
    return tuple(slice(0, int(dim)) for dim in shape)


def _timepoint_region(shape: tuple[int, ...], timepoint: int) -> tuple[slice, ...]:
    """Return the array region selecting one timepoint in a full label array."""
    return (slice(int(timepoint), int(timepoint) + 1),) + tuple(slice(0, int(dim)) for dim in shape[1:])


def _persist_written_timepoints(label_group, timepoints: Any) -> None:
    """Merge newly written timepoints into the label group's microio metadata."""
    microio = dict(label_group.attrs.asdict().get("microio", {}))
    current = {int(item) for item in microio.get("written_timepoints", [])}
    current.update(int(item) for item in timepoints)
    microio["written_timepoints"] = sorted(current)
    label_group.attrs["microio"] = microio


def _validate_existing_timepoint_write(label_group, *, timepoint: int, overwrite_timepoint: bool, initialized: bool) -> None:
    """Apply overwrite protection for writes into an existing label image."""
    if initialized:
        return
    microio = dict(label_group.attrs.asdict().get("microio", {}))
    written = {int(item) for item in microio.get("written_timepoints", [])}
    if int(timepoint) in written and not overwrite_timepoint:
        raise FileExistsError(
            f"Timepoint {timepoint} for {label_group.path} already exists; pass overwrite_timepoint=True to replace it."
        )


def _label_extra_attrs(
    scene_id: str,
    source_level_path: str,
    attrs: dict[str, Any] | None,
    *,
    channel_mode: str,
    write_mode: str,
    written_timepoints: list[int],
) -> dict[str, Any]:
    """Build the non-OME microio metadata block for a label image group."""
    microio = {
        "schema": MICROIO_WRITER_LABEL_SCHEMA,
        "schema_version": MICROIO_WRITER_LABEL_SCHEMA_VERSION,
        "source_scene_id": scene_id,
        "source_level": str(source_level_path),
        "channel_mode": channel_mode,
        "write_mode": write_mode,
        "written_timepoints": [int(item) for item in written_timepoints],
    }
    if attrs:
        microio["label-attrs"] = deepcopy(attrs)
    return {"microio": microio}


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
        if step != 1:
            raise NotImplementedError(f"Axis {axis!r} slice step {step!r} is not implemented for ROI writes")
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
