"""Shared multiscale-reader helpers for scenes and label images."""

from __future__ import annotations

import logging
from typing import Any

import dask.array as da

from microio.common.constants import AXES_ORDER
from microio.common.models import LevelRef


logger = logging.getLogger("microio.reader.multiscale")


def validate_multiscale_axes(container_id: str, multiscale: dict) -> None:
    """Validate axis metadata against the library's supported axis contract."""
    logger.debug("Validating multiscale axes for %s", container_id)
    axes = multiscale.get("axes")
    datasets = multiscale.get("datasets")
    if not isinstance(axes, list) or not axes:
        raise ValueError(f"{container_id} has no multiscale axes metadata")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"{container_id} has no multiscale datasets metadata")

    axis_names = [axis.get("name") for axis in axes]
    if any(not isinstance(name, str) or not name for name in axis_names):
        raise ValueError(f"{container_id} has unnamed multiscale axes")
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(f"{container_id} has duplicate multiscale axes: {axis_names}")
    if tuple(axis_names) != AXES_ORDER:
        raise ValueError(f"{container_id} uses unsupported axis order {axis_names}; expected {list(AXES_ORDER)}")
    logger.debug("Validated multiscale axes for %s: %s", container_id, axis_names)


def list_container_levels(
    *,
    ds,
    cache_key: tuple[str, ...],
    scene_id: str,
    container_kind: str,
    container_name: str | None,
    container_id: str,
    group: Any,
    multiscale: dict,
) -> list[LevelRef]:
    """Enumerate and validate all multiscale levels for one container."""
    cached = ds._level_refs_cache.get(cache_key)
    if cached is not None:
        logger.debug("Using cached level refs for %s", container_id)
        return cached

    logger.debug("Resolving level refs for %s", container_id)
    axes = tuple(axis["name"] for axis in multiscale["axes"])
    axis_units = tuple(axis.get("unit") for axis in multiscale["axes"])
    levels: list[LevelRef] = []
    previous_shape: tuple[int, ...] | None = None

    for level_index, dataset in enumerate(multiscale["datasets"]):
        path = str(dataset.get("path"))
        if path not in group:
            raise ValueError(f"{container_id} level {path!r} is listed in multiscales but missing from the Zarr group")
        transforms = dataset.get("coordinateTransformations")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError(f"{container_id} level {path!r} has no coordinate transformations")
        scale = transforms[0].get("scale")
        if not isinstance(scale, list):
            raise ValueError(f"{container_id} level {path!r} has malformed scale metadata")
        if len(scale) != len(axes):
            raise ValueError(
                f"{container_id} level {path!r} scale length {len(scale)} does not match axis count {len(axes)}"
            )
        array = group[path]
        shape = tuple(int(dim) for dim in array.shape)
        if len(shape) != len(axes):
            raise ValueError(f"{container_id} level {path!r} ndim {len(shape)} does not match multiscale axis count {len(axes)}")
        if previous_shape is not None:
            validate_pyramid_shapes(container_id, path, previous_shape, shape, axes)
        previous_shape = shape
        levels.append(
            LevelRef(
                scene_id=scene_id,
                level_index=level_index,
                path=path,
                shape=shape,
                dtype=str(array.dtype),
                scale=tuple(float(value) for value in scale),
                axis_names=axes,
                axis_units=axis_units,
                container_kind=container_kind,
                container_name=container_name,
            )
        )
        logger.debug(
            "Validated %s level %s with shape=%s dtype=%s scale=%s",
            container_id,
            path,
            shape,
            array.dtype,
            scale,
        )

    ds._level_refs_cache[cache_key] = levels
    logger.debug("Cached %d level refs for %s", len(levels), container_id)
    return levels


def resolve_container_level(levels: list[LevelRef], level: int | str, *, container_id: str) -> LevelRef:
    """Resolve one level by index or path within a container."""
    if isinstance(level, int):
        if 0 <= level < len(levels):
            logger.debug("Resolved %s level index %s to path %s", container_id, level, levels[level].path)
            return levels[level]
        raise KeyError(f"{container_id} level index {level} is out of range; available=0..{len(levels) - 1}")

    level_text = str(level)
    for candidate in levels:
        if candidate.path == level_text:
            logger.debug("Resolved %s level path %s", container_id, level_text)
            return candidate
    raise KeyError(f"{container_id} has no level path {level_text!r}; available={[item.path for item in levels]}")


def read_container_level(group: Any, level_info: LevelRef):
    """Read one image level as the underlying Zarr array."""
    logger.debug(
        "Opening %s container level %s for scene %s",
        level_info.container_kind,
        level_info.path,
        level_info.scene_id,
    )
    return group[level_info.path]


def read_container_level_numpy(group: Any, level_info: LevelRef):
    """Materialize one image level eagerly as a NumPy array."""
    return read_container_level(group, level_info)[:]


def wrap_zarr_as_dask(array):
    """Wrap a Zarr-backed array as Dask while preserving existing chunking."""
    chunks = getattr(array, "chunks", None)
    logger.debug("Wrapping array with chunks=%s as Dask", chunks)
    if chunks is not None:
        return da.from_array(array, chunks=chunks, inline_array=True)
    return da.from_array(array, inline_array=True)


def validate_pyramid_shapes(
    container_id: str,
    level_path: str,
    previous_shape: tuple[int, ...],
    current_shape: tuple[int, ...],
    axes: tuple[str, ...],
) -> None:
    """Ensure coarser pyramid levels do not grow relative to prior levels."""
    for axis_name, previous_dim, current_dim in zip(axes, previous_shape, current_shape):
        if current_dim <= 0:
            raise ValueError(f"{container_id} level {level_path!r} axis {axis_name} has invalid size {current_dim}")
        if current_dim > previous_dim:
            raise ValueError(
                f"{container_id} level {level_path!r} axis {axis_name} grows from {previous_dim} to {current_dim}"
            )
