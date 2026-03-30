"""Workspace helpers for computation-friendly copies of source scenes."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import logging
from pathlib import Path
import shutil
from typing import Any
import xml.etree.ElementTree as ET

import numpy as np
import zarr

from microio.common.constants import MICROIO_WORKSPACE_SCHEMA, MICROIO_WORKSPACE_SCHEMA_VERSION
from microio.common.models import DatasetHandle, WorkspaceHandle
from microio.common.ngff import node_zarr_format, non_ome_attrs, ome_metadata, replace_ome_attrs
from microio.reader.labels import list_labels as list_scene_labels
from microio.reader.open import open_dataset
from microio.reader.ome_xml import NS, OME_NS
from microio.writer.common import default_chunks, replace_node_ome_metadata, write_array
from microio.writer.images import (
    _image_label_version,
    _single_scale_multiscale,
)


logger = logging.getLogger("microio.workspace")


def create_workspace(
    ds: DatasetHandle,
    destination: str | Path,
    scene: int | str,
    *,
    source_level: int | str = 0,
    chunks: tuple[int, ...] | None = None,
    labels: list[str] | None = None,
    overwrite: bool = False,
    threads: int | None = None,
) -> WorkspaceHandle:
    """Create a single-scene computation workspace as a sibling Zarr store."""
    ref = ds.scene_ref(scene)
    level = ds.level_ref(ref.id, source_level)
    source = ds.read_level(ref.id, level.path)
    requested_labels = _normalize_workspace_labels(ds, ref.id, labels)
    destination_path = Path(destination)
    if destination_path.exists():
        if not overwrite:
            raise FileExistsError(f"Workspace path already exists: {destination_path}")
        shutil.rmtree(destination_path)

    zarr_format = node_zarr_format(ds.root)
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    root = zarr.open(destination_path, mode="w", zarr_format=zarr_format)
    _write_workspace_root_metadata(ds, root, ref.id, level, chunks=chunks, labels=requested_labels)
    _write_workspace_ome_group(ds, root, destination_path, ref.id)
    _write_workspace_scene(ds, root, ref.id, level.path, source, chunks=chunks, threads=threads)
    if requested_labels:
        _carry_labels(ds, root, ref.id, level.path, requested_labels, chunks=chunks, threads=threads)
    workspace_ds = open_dataset(destination_path, mode="a")
    return open_workspace(workspace_ds)


def open_workspace(ds: DatasetHandle) -> WorkspaceHandle:
    """Validate that ``ds`` is a computation workspace and return its metadata."""
    metadata = _workspace_metadata(ds.root)
    if metadata is None:
        raise ValueError(f"Dataset {ds.path} is not a microio computation workspace")
    return WorkspaceHandle(
        workspace_path=ds.path,
        source_dataset_path=Path(metadata["source_dataset_path"]),
        source_scene_id=str(metadata["source_scene_id"]),
        source_level=str(metadata["source_level"]),
        source_level_path=str(metadata["source_level_path"]),
        source_level_shape=tuple(int(dim) for dim in metadata["source_level_shape"]),
        source_level_scale=tuple(float(value) for value in metadata["source_level_scale"]),
        chunks=tuple(int(chunk) for chunk in metadata["chunks"]),
        carried_labels=tuple(str(name) for name in metadata.get("carried_labels", [])),
        source_scene_name=metadata.get("source_scene_name"),
        created_at=metadata.get("created_at"),
    )


def delete_workspace(ds: DatasetHandle) -> Path:
    """Delete the current workspace after provenance validation."""
    workspace = open_workspace(ds)
    path = workspace.workspace_path
    shutil.rmtree(path)
    return path


def commit_workspace_labels(
    ds: DatasetHandle,
    name: str,
    data: Any | None = None,
    *,
    workspace_label: str | None = None,
    chunks: tuple[int, ...] | None = None,
    dtype: Any | None = None,
    attrs: dict[str, Any] | None = None,
    colors: list[dict[str, Any]] | None = None,
    properties: list[dict[str, Any]] | None = None,
    overwrite: bool = False,
    timepoint: int | None = None,
    overwrite_timepoint: bool = False,
    threads: int | None = None,
):
    """Commit a computed label image from a workspace to its source dataset."""
    workspace = open_workspace(ds)
    if workspace.source_level != "0":
        raise ValueError(
            "Workspace label commit currently requires source_level=0 because microio label writes "
            "must remain aligned to source level 0."
        )

    payload = data
    candidate = str(workspace_label or name)
    if payload is None:
        _ensure_workspace_label_can_commit(ds, workspace, candidate)
        payload = ds.read_label(workspace.source_scene_id, candidate, 0)

    resolved_attrs = attrs
    resolved_colors = colors
    resolved_properties = properties
    if workspace_label is not None:
        metadata = _workspace_label_commit_metadata(ds, workspace.source_scene_id, candidate)
        if resolved_attrs is None:
            resolved_attrs = metadata.label_attrs
        if resolved_colors is None:
            resolved_colors = metadata.colors
        if resolved_properties is None:
            resolved_properties = metadata.properties

    target_ds = open_dataset(workspace.source_dataset_path, mode="a")
    if timepoint is None:
        return target_ds.write_label_image(
            workspace.source_scene_id,
            name,
            payload,
            source_level=workspace.source_level,
            chunks=chunks,
            dtype=dtype,
            attrs=resolved_attrs,
            colors=resolved_colors,
            properties=resolved_properties,
            overwrite=overwrite,
            threads=threads,
        )
    return target_ds.write_label_timepoint(
        workspace.source_scene_id,
        name,
        payload,
        timepoint=timepoint,
        source_level=workspace.source_level,
        chunks=chunks,
        dtype=dtype,
        attrs=resolved_attrs,
        colors=resolved_colors,
        properties=resolved_properties,
        overwrite=overwrite,
        overwrite_timepoint=overwrite_timepoint,
        threads=threads,
    )


def commit_workspace_table(
    ds: DatasetHandle,
    name: str,
    data: Any | None = None,
    *,
    workspace_table: str | None = None,
    attrs: dict[str, Any] | None = None,
    overwrite: bool = False,
    append: bool = False,
    chunk_length: int | None = None,
):
    """Commit a table from a workspace to its source dataset."""
    workspace = open_workspace(ds)
    payload = data
    candidate = str(workspace_table or name)
    if payload is None:
        payload = ds.load_table(workspace.source_scene_id, candidate)
    resolved_attrs = attrs
    if workspace_table is not None and resolved_attrs is None:
        resolved_attrs = ds.read_table(workspace.source_scene_id, candidate).table_attrs
    target_ds = open_dataset(workspace.source_dataset_path, mode="a")
    return target_ds.write_table(
        workspace.source_scene_id,
        name,
        payload,
        attrs=resolved_attrs,
        overwrite=overwrite,
        append=append,
        chunk_length=chunk_length,
    )


def _normalize_workspace_labels(ds: DatasetHandle, scene_id: str, labels: list[str] | None) -> list[str]:
    if labels is None:
        return []
    requested = [str(name) for name in labels]
    available = set(list_scene_labels(ds, scene_id))
    missing = [name for name in requested if name not in available]
    if missing:
        raise KeyError(f"Unknown label names for scene {scene_id}: {missing}; available={sorted(available)}")
    return requested


def _write_workspace_root_metadata(
    ds: DatasetHandle,
    root,
    scene_id: str,
    level,
    *,
    chunks: tuple[int, ...] | None,
    labels: list[str],
) -> None:
    extra = non_ome_attrs(ds.root)
    microio = dict(extra.get("microio", {}))
    microio["workspace"] = {
        "schema": MICROIO_WORKSPACE_SCHEMA,
        "schema_version": MICROIO_WORKSPACE_SCHEMA_VERSION,
        "source_dataset_path": str(ds.path.resolve()),
        "source_scene_id": str(scene_id),
        "source_scene_name": ds.scene_ref(scene_id).name,
        "source_level": str(level.level_index),
        "source_level_path": str(level.path),
        "source_level_shape": [int(dim) for dim in level.shape],
        "source_level_scale": [float(value) for value in level.scale],
        "chunks": [int(chunk) for chunk in _resolved_chunks(level.shape, level.dtype, chunks)],
        "carried_labels": [str(name) for name in labels],
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    extra["microio"] = microio
    replace_ome_attrs(root, ome_metadata(ds.root), extra_attrs=extra)


def _write_workspace_ome_group(ds: DatasetHandle, root, destination_path: Path, scene_id: str) -> None:
    ome_source = ds.root.get("OME")
    if ome_source is None:
        return
    ome_group = root.create_group("OME")
    ome_attrs = ome_metadata(ome_source)
    ome_attrs["series"] = [str(scene_id)]
    replace_ome_attrs(ome_group, ome_attrs, extra_attrs=non_ome_attrs(ome_source))
    xml_path = destination_path / "OME" / "METADATA.ome.xml"
    xml_path.write_text(_subset_ome_xml(ds, scene_id), encoding="utf-8")


def _write_workspace_scene(
    ds: DatasetHandle,
    root,
    scene_id: str,
    source_level_path: str,
    source,
    *,
    chunks: tuple[int, ...] | None,
    threads: int | None,
) -> None:
    scene_group = root.create_group(scene_id)
    attrs = ds.read_scene_metadata(scene_id)
    attrs["multiscales"] = [_single_scene_level_multiscale(ds, scene_id, source_level_path)]
    scene_ome = ome_metadata(ds.root[scene_id])
    scene_ome["multiscales"] = [deepcopy(attrs["multiscales"][0])]
    if "omero" in attrs:
        scene_ome["omero"] = deepcopy(attrs["omero"])
    replace_ome_attrs(scene_group, scene_ome, extra_attrs=non_ome_attrs(ds.root[scene_id]))
    target_chunks = _resolved_chunks(source.shape, source.dtype, chunks)
    write_array(
        scene_group,
        "0",
        source,
        chunks=target_chunks,
        threads=threads,
        dimension_names=tuple(axis["name"] for axis in attrs["multiscales"][0]["axes"]) if node_zarr_format(scene_group) >= 3 else None,
    )


def _carry_labels(
    ds: DatasetHandle,
    root,
    scene_id: str,
    source_level_path: str,
    labels: list[str],
    *,
    chunks: tuple[int, ...] | None,
    threads: int | None,
) -> None:
    scene_group = root[scene_id]
    labels_group = scene_group.create_group("labels")
    replace_node_ome_metadata(labels_group, {"labels": sorted(labels)})
    for name in labels:
        source_group = ds.root[scene_id]["labels"][name]
        label_level = ds.label_level_ref(scene_id, name, source_level_path)
        label_data = ds.read_label(scene_id, name, label_level.path)
        target_group = labels_group.create_group(name)
        target_chunks = _resolved_chunks(label_level.shape, label_level.dtype, chunks)
        write_array(
            target_group,
            "0",
            label_data,
            chunks=target_chunks,
            threads=threads,
            dimension_names=label_level.axis_names if node_zarr_format(target_group) >= 3 else None,
        )
        label_ome = ome_metadata(source_group)
        label_multiscale = _single_label_level_multiscale(ds, scene_id, name, label_level.path)
        label_ome["multiscales"] = [label_multiscale]
        image_label = deepcopy(label_ome.get("image-label", {}))
        image_label["source"] = {"image": "../../"}
        image_label["version"] = _image_label_version(label_multiscale)
        label_ome["image-label"] = image_label
        extra = non_ome_attrs(source_group)
        microio = dict(extra.get("microio", {}))
        workspace_md = dict(microio.get("workspace", {}))
        workspace_md.update({"read_only": True, "carried_from_source": True})
        microio["workspace"] = workspace_md
        extra["microio"] = microio
        replace_ome_attrs(target_group, label_ome, extra_attrs=extra)


def _single_scene_level_multiscale(ds: DatasetHandle, scene_id: str, source_level_path: str) -> dict[str, Any]:
    multiscale = deepcopy(ds.read_multiscale_metadata(scene_id))
    dataset_md = next(item for item in multiscale["datasets"] if str(item.get("path")) == str(source_level_path))
    return _single_scale_multiscale(multiscale, dataset_md, multiscale.get("name") or scene_id)


def _single_label_level_multiscale(ds: DatasetHandle, scene_id: str, name: str, source_level_path: str) -> dict[str, Any]:
    source_group = ds.root[scene_id]["labels"][name]
    attrs = source_group.attrs.asdict()
    source_ome = attrs.get("ome", attrs)
    multiscale = deepcopy(source_ome["multiscales"][0])
    dataset_md = next(item for item in multiscale["datasets"] if str(item.get("path")) == str(source_level_path))
    return _single_scale_multiscale(multiscale, dataset_md, multiscale.get("name") or name)


def _workspace_metadata(root) -> dict[str, Any] | None:
    attrs = root.attrs.asdict()
    microio = attrs.get("microio", {})
    workspace = microio.get("workspace")
    return workspace if isinstance(workspace, dict) else None


def _resolved_chunks(shape: tuple[int, ...], dtype: Any, chunks: tuple[int, ...] | None) -> tuple[int, ...]:
    return default_chunks(tuple(int(dim) for dim in shape), np.dtype(dtype), chunks)


def _subset_ome_xml(ds: DatasetHandle, scene_id: str) -> str:
    xml_text = ds.read_ome_xml()
    root = ET.fromstring(xml_text)
    images = root.findall("ome:Image", NS)
    keep = ds.read_scene_ome_metadata(scene_id).index
    for index, image in enumerate(list(images)):
        if index != keep:
            root.remove(image)
    ET.register_namespace("", OME_NS)
    return ET.tostring(root, encoding="unicode")


def _ensure_workspace_label_can_commit(ds: DatasetHandle, workspace: WorkspaceHandle, name: str) -> None:
    if name not in ds.list_labels(workspace.source_scene_id):
        raise KeyError(f"Workspace scene {workspace.source_scene_id} has no label named {name!r}")
    metadata = ds.read_label_metadata(workspace.source_scene_id, name)
    workspace_md = metadata.microio.get("workspace", {})
    if workspace_md.get("read_only") or workspace_md.get("carried_from_source"):
        raise PermissionError(
            f"Workspace label {name!r} is a carried read-only source label and cannot be committed as a computed output."
        )


def _workspace_label_commit_metadata(ds: DatasetHandle, scene_id: str, name: str):
    """Read logical label metadata in the same shape used by the writer API."""
    return ds.read_label_metadata(scene_id, name)
