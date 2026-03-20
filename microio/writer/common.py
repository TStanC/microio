"""Shared helpers for constrained write-side enrichment."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import logging
from math import prod
from typing import Any

import dask
import dask.array as da
import numpy as np

from microio.common.constants import ALLOWED_SCENE_WRITE_GROUPS
from microio.common.mutations import require_writable
from microio.common.ngff import node_zarr_format, non_ome_attrs, ome_metadata, replace_ome_attrs
from microio.reader.metadata import level_ref, multiscale_metadata, scene_ref


logger = logging.getLogger("microio.writer.common")


def require_writeable_scene(ds, scene: int | str):
    """Resolve one writable scene and reject unsupported scene layouts."""
    require_writable(ds)
    ref = scene_ref(ds, scene)
    logger.debug("Resolved writable scene %s", ref.id)
    _validate_scene_children(ds, ref.id)
    return ref


def _validate_scene_children(ds, scene_id: str) -> None:
    """Reject scene layouts that contain unsupported non-image child groups."""
    scene = ds.root[scene_id]
    invalid = [
        name
        for name, _ in scene.groups()
        if not name.isdigit() and name not in ALLOWED_SCENE_WRITE_GROUPS
    ]
    if invalid:
        logger.warning("Scene %s contains unsupported write-side children: %s", scene_id, sorted(invalid))
        raise ValueError(
            f"Scene {scene_id} contains unsupported subgroups for writer operations: {sorted(invalid)}"
        )


def source_level_metadata(ds, scene: int | str, source_level: int | str) -> tuple[Any, dict, Any]:
    """Load source-level metadata used to seed label and ROI writes."""
    ref = scene_ref(ds, scene)
    level = level_ref(ds, ref.id, source_level)
    ms = deepcopy(multiscale_metadata(ds, ref.id))
    dataset_md = next(item for item in ms["datasets"] if str(item.get("path")) == level.path)
    logger.debug("Loaded source level metadata for scene %s level %s", ref.id, level.path)
    return ref, ms, dataset_md


def source_pyramid_metadata(ds, scene: int | str) -> tuple[Any, dict, list[Any]]:
    """Load source-scene metadata used to seed full label pyramids."""
    ref = scene_ref(ds, scene)
    ms = deepcopy(multiscale_metadata(ds, ref.id))
    datasets = [deepcopy(item) for item in ms["datasets"]]
    logger.debug("Loaded source pyramid metadata for scene %s with %d levels", ref.id, len(datasets))
    return ref, ms, datasets


def default_chunks(shape: tuple[int, ...], dtype: np.dtype, chunks: tuple[int, ...] | None) -> tuple[int, ...]:
    """Choose a conservative chunk layout for newly written arrays."""
    if chunks is not None:
        return tuple(int(max(1, min(dim, chunk))) for dim, chunk in zip(shape, chunks, strict=True))
    itemsize = max(1, int(dtype.itemsize))
    target_bytes = 8 * 1024 * 1024
    target_items = max(1, target_bytes // itemsize)
    resolved = list(shape)
    while prod(resolved) > target_items:
        idx = max(range(len(resolved)), key=lambda i: resolved[i])
        if resolved[idx] == 1:
            break
        resolved[idx] = max(1, resolved[idx] // 2)
    return tuple(int(max(1, min(dim, chunk))) for dim, chunk in zip(shape, resolved, strict=True))


def ensure_group_absent_or_overwrite(parent, name: str, *, overwrite: bool) -> None:
    """Guard a child name against accidental replacement."""
    if name not in parent:
        return
    if not overwrite:
        logger.info("Refusing to overwrite existing group %s/%s", parent.path, name)
        raise FileExistsError(f"{parent.path}/{name} already exists; pass overwrite=True to replace it.")
    logger.info("Overwriting existing group %s/%s", parent.path, name)
    del parent[name]


def require_child_group(parent, name: str):
    """Return an existing child group or create it if absent."""
    if name in parent:
        return parent[name]
    logger.debug("Creating child group %s/%s", parent.path, name)
    return parent.create_group(name)


def coerce_array(data: Any):
    """Normalize array-like input into Dask or NumPy form."""
    if isinstance(data, da.Array):
        return data
    if hasattr(data, "shape") and hasattr(data, "dtype") and hasattr(data, "__getitem__"):
        try:
            return da.from_array(data, chunks=getattr(data, "chunks", "auto"), inline_array=True)
        except Exception:
            pass
    return np.asarray(data)


def maybe_cast_array(data: Any, dtype: Any | None):
    """Apply a dtype cast only when explicitly requested and value-safe."""
    if dtype is None:
        return data
    if isinstance(data, da.Array):
        return data.astype(dtype)
    arr = np.asarray(data)
    if arr.dtype == np.dtype(dtype):
        return arr
    cast = arr.astype(dtype)
    if not np.array_equal(arr, cast.astype(arr.dtype, copy=False)):
        raise ValueError(f"Requested dtype {dtype!r} would change array values")
    return cast


def write_array(
    group,
    name: str,
    data: Any,
    *,
    chunks: tuple[int, ...],
    threads: int | None = None,
    dimension_names: tuple[str, ...] | None = None,
):
    """Persist one array using either Dask or NumPy-backed write paths."""
    dtype = np.dtype(data.dtype) if hasattr(data, "dtype") else np.asarray(data).dtype
    target = group.create_array(
        name,
        shape=tuple(int(dim) for dim in data.shape),
        dtype=dtype,
        chunks=chunks,
        dimension_names=dimension_names,
        overwrite=True,
        write_data=False,
    )
    if isinstance(data, da.Array):
        logger.debug("Writing Dask array %s/%s with chunks=%s threads=%s", group.path, name, chunks, threads or 1)
        with dask.config.set(scheduler="threads", num_workers=threads or 1):
            da.store(data.rechunk(chunks), target, lock=False, compute=True)
        return target

    arr = np.asarray(data)
    logger.debug("Writing NumPy array %s/%s with chunks=%s threads=%s", group.path, name, chunks, threads)
    if threads is None or threads <= 1 or arr.ndim == 0:
        target[...] = arr
        return target

    ranges = [(start, min(start + chunks[0], arr.shape[0])) for start in range(0, arr.shape[0], chunks[0])]

    def _write_chunk(start: int, stop: int) -> None:
        index = [slice(None)] * arr.ndim
        index[0] = slice(start, stop)
        target[tuple(index)] = arr[tuple(index)]

    with ThreadPoolExecutor(max_workers=threads) as pool:
        list(pool.map(lambda bounds: _write_chunk(*bounds), ranges))
    return target


def normalize_slice_spec(spec: Any) -> slice | int:
    """Accept the ROI slice forms supported by the public writer API."""
    if isinstance(spec, slice):
        return spec
    if isinstance(spec, tuple) and len(spec) == 2:
        return slice(spec[0], spec[1], None)
    if isinstance(spec, int):
        return spec
    raise TypeError(f"Unsupported slice spec: {spec!r}")


def replace_node_ome_metadata(node, ome: dict[str, Any], *, extra_attrs: dict[str, Any] | None = None) -> None:
    """Persist semantic OME metadata in a format-compatible attrs layout."""
    merged_extra = non_ome_attrs(node)
    if extra_attrs:
        merged_extra.update(deepcopy(extra_attrs))
    replace_ome_attrs(node, ome, extra_attrs=merged_extra)


def read_node_ome_metadata(node) -> dict[str, Any]:
    """Return semantic OME metadata for a group or array."""
    return ome_metadata(node)


def group_zarr_format(node) -> int:
    """Return a group's Zarr format number."""
    return node_zarr_format(node)
