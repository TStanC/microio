"""Reader helpers for NGFF label image groups."""

from __future__ import annotations

from copy import deepcopy
import logging

from microio.common.models import DatasetHandle, LabelReadResult, LevelRef
from microio.common.ngff import flattened_attrs, ome_metadata
from microio.reader.multiscale import (
    list_container_levels,
    read_container_level,
    read_container_level_numpy,
    resolve_container_level,
    validate_multiscale_axes,
    wrap_zarr_as_dask,
)


logger = logging.getLogger("microio.reader.labels")


def list_labels(ds: DatasetHandle, scene: int | str) -> list[str]:
    """List label image names stored under one scene."""
    ref = ds.scene_ref(scene)
    scene_group = ds.root[ref.id]
    if "labels" not in scene_group:
        logger.debug("Scene %s has no labels group", ref.id)
        return []
    names = sorted(str(name) for name, _ in scene_group["labels"].groups())
    logger.debug("Found %d label groups for scene %s: %s", len(names), ref.id, names)
    return names


def read_label_metadata(ds: DatasetHandle, scene: int | str, name: str) -> LabelReadResult:
    """Read label-group metadata without loading image data.

    The returned :class:`LabelReadResult` preserves the raw metadata views
    while also exposing the logical ``label_attrs``, ``colors``, and
    ``properties`` fields corresponding to the label-writer API.
    """
    ref = ds.scene_ref(scene)
    logger.debug("Reading label metadata for scene %s label %s", ref.id, name)
    labels_group = _labels_group(ds, ref.id)
    label_group = _label_group(ds, ref.id, name)
    attrs = flattened_attrs(label_group)
    multiscale = multiscale_metadata(ds, ref.id, name)
    microio = dict(attrs.get("microio", {}))
    ome = ome_metadata(label_group)
    image_label = ome.get("image-label", {}) if isinstance(ome, dict) else {}
    return LabelReadResult(
        scene_id=ref.id,
        label_name=str(name),
        attrs={**deepcopy(attrs), "multiscales": [deepcopy(multiscale)]},
        microio=microio,
        label_attrs=deepcopy(microio.get("label-attrs")) if isinstance(microio.get("label-attrs"), dict) else None,
        colors=deepcopy(image_label.get("colors")) if isinstance(image_label.get("colors"), list) else None,
        properties=deepcopy(image_label.get("properties")) if isinstance(image_label.get("properties"), list) else None,
        ome=ome,
        group_attrs=flattened_attrs(label_group),
        labels_group_attrs=flattened_attrs(labels_group),
    )


def multiscale_metadata(ds: DatasetHandle, scene: int | str, name: str) -> dict:
    """Return the validated primary multiscales block for one label image."""
    ref = ds.scene_ref(scene)
    label_group = _label_group(ds, ref.id, name)
    attrs = flattened_attrs(label_group)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"Label {name!r} for scene {ref.id} has no multiscales metadata")
    multiscale = multiscales[0]
    validate_multiscale_axes(f"Label {name!r} for scene {ref.id}", multiscale)
    logger.debug("Validated label multiscales metadata for scene %s label %s", ref.id, name)
    return multiscale


def list_label_levels(ds: DatasetHandle, scene: int | str, name: str) -> list[LevelRef]:
    """Enumerate and validate all multiscale levels for one label image."""
    ref = ds.scene_ref(scene)
    label_group = _label_group(ds, ref.id, name)
    multiscale = multiscale_metadata(ds, ref.id, name)
    return list_container_levels(
        ds=ds,
        cache_key=("label", ref.id, str(name)),
        scene_id=ref.id,
        container_kind="label",
        container_name=str(name),
        container_id=f"Label {name!r} for scene {ref.id}",
        group=label_group,
        multiscale=multiscale,
    )


def label_level_ref(ds: DatasetHandle, scene: int | str, name: str, level: int | str) -> LevelRef:
    """Resolve one label level by index or path."""
    ref = ds.scene_ref(scene)
    return resolve_container_level(list_label_levels(ds, ref.id, name), level, container_id=f"Label {name!r} for scene {ref.id}")


def read_label(ds: DatasetHandle, scene: int | str, name: str, level: int | str = 0):
    """Read one label-image level as a lazy Dask array."""
    logger.debug("Reading label level lazily for scene=%s label=%s level=%s", scene, name, level)
    return wrap_zarr_as_dask(read_label_zarr(ds, scene, name, level))


def read_label_zarr(ds: DatasetHandle, scene: int | str, name: str, level: int | str = 0):
    """Read one label-image level as the underlying Zarr array."""
    ref = ds.scene_ref(scene)
    label_group = _label_group(ds, ref.id, name)
    level_info = label_level_ref(ds, ref.id, name, level)
    logger.debug("Opening label array for scene %s label %s level %s", ref.id, name, level_info.path)
    return read_container_level(label_group, level_info)


def read_label_numpy(ds: DatasetHandle, scene: int | str, name: str, level: int | str = 0):
    """Materialize one label-image level eagerly as a NumPy array."""
    ref = ds.scene_ref(scene)
    label_group = _label_group(ds, ref.id, name)
    level_info = label_level_ref(ds, ref.id, name, level)
    logger.debug("Materializing label array for scene %s label %s level %s", ref.id, name, level_info.path)
    return read_container_level_numpy(label_group, level_info)


def _labels_group(ds: DatasetHandle, scene_id: str):
    scene_group = ds.root[scene_id]
    if "labels" not in scene_group:
        logger.warning("Requested labels for scene %s but no labels group exists", scene_id)
        raise KeyError(f"Scene {scene_id} has no labels; available=[]")
    return scene_group["labels"]


def _label_group(ds: DatasetHandle, scene: int | str, name: str):
    ref = ds.scene_ref(scene)
    labels_group = ds.root[ref.id].get("labels")
    if labels_group is None or str(name) not in labels_group:
        logger.warning("Requested missing label %s for scene %s", name, ref.id)
        raise KeyError(f"Scene {ref.id} has no label named {name!r}; available={list_labels(ds, ref.id)}")
    return labels_group[str(name)]
