"""NGFF-specific writing helpers for scenes and root metadata."""

from __future__ import annotations

import logging
from pathlib import Path
import math
import shutil

import numpy as np
import zarr
from ome_zarr.io import parse_url
from ome_zarr.writer import add_metadata, write_image

from .xmlparse import SceneXmlMeta


logger = logging.getLogger("microio.writer.ngff")


def create_root_store(output_path: Path, overwrite: bool = False):
    """Create a writable Zarr root group at the requested output path.

    Parameters
    ----------
    output_path:
        Filesystem path where the OME-Zarr hierarchy will be created.
    overwrite:
        When ``True``, an existing directory is removed before the new store is
        created.
    """
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists: {output_path}")
    if output_path.exists() and overwrite:
        logger.warning("Removing existing output store before rewrite: %s", output_path)
        shutil.rmtree(output_path)
    store = parse_url(str(output_path), mode="w").store
    logger.info("Created root OME-Zarr store at %s", output_path)
    return zarr.group(store=store)


def write_scene_image(
    scene_group,
    data,
    scene: SceneXmlMeta,
    axis_scale: list[float],
    axis_units: list[str | None],
    *,
    ngff_version: str = "0.5",
    chunks: tuple[int, int, int, int, int] | None = None,
) -> None:
    """Write one scene image pyramid level and NGFF scene metadata.

    Parameters
    ----------
    scene_group:
        Destination scene group under the root OME-Zarr store.
    data:
        Array-like image data, expected by the current writer pipeline in
        ``TCZYX`` order.
    scene:
        Scene metadata used to derive OME-Zarr channel labels and colors.
    axis_scale:
        Physical scale for the ``t``, ``c``, ``z``, ``y``, and ``x`` axes.
    axis_units:
        Unit strings aligned with ``axis_scale``.
    ngff_version:
        Target NGFF metadata version string.
    chunks:
        Optional chunk shape passed through to the OME-Zarr writer.
    """
    logger.info(
        "Writing scene image %s with shape=%s dtype=%s chunks=%s",
        scene_group.name,
        getattr(data, "shape", None),
        getattr(data, "dtype", None),
        chunks,
    )
    axes = [
        {"name": "t", "type": "time", "unit": axis_units[0]},
        {"name": "c", "type": "channel", "unit": None},
        {"name": "z", "type": "space", "unit": axis_units[2]},
        {"name": "y", "type": "space", "unit": axis_units[3]},
        {"name": "x", "type": "space", "unit": axis_units[4]},
    ]
    ctrans = [[{"type": "scale", "scale": axis_scale}]]

    write_image(
        image=data,
        group=scene_group,
        axes=axes,
        scaler=None,
        storage_options={"chunks": chunks} if chunks else None,
        coordinate_transformations=ctrans,
    )

    omero = {
        "name": scene_group.name.split("/")[-1],
        "channels": _build_omero_channels(data, scene),
    }
    add_metadata(scene_group, {"omero": omero})
    scene_group.attrs["ome"] = dict(scene_group.attrs.asdict().get("ome", {}), version=ngff_version)
    logger.debug("Finished writing scene metadata for %s", scene_group.name)


def write_root_ome_group(
    root: zarr.Group,
    scenes: list[str],
    source_name: str,
    source_location: str,
    ome_xml: str,
    acquisition_software: str | None = None,
    *,
    ngff_version: str = "0.5",
) -> None:
    """Write root-level OME-Zarr metadata and the ``OME/METADATA.ome.xml`` sidecar."""
    root.attrs["ome"] = {"version": ngff_version, "bioformats2raw.layout": 3}

    ome_group = root.require_group("OME")
    ome_group.attrs["scenes"] = [{"id": s, "path": s} for s in scenes]
    ome_group.attrs["originalName"] = source_name
    ome_group.attrs["originalLocation"] = source_location
    if acquisition_software:
        ome_group.attrs["acquisitionSoftware"] = acquisition_software
    ome_group.attrs["ome"] = {"version": ngff_version, "series": scenes}

    metadata_xml_path = Path(ome_group.store.root) / ome_group.path / "METADATA.ome.xml"
    metadata_xml_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_xml_path.write_text(ome_xml, encoding="utf-8")
    logger.info("Wrote OME sidecar XML for %d scenes to %s", len(scenes), metadata_xml_path)


def _build_omero_channels(data, scene: SceneXmlMeta) -> list[dict]:
    """Build OME-Zarr ``omero.channels`` metadata for one scene."""
    channel_count = int(data.shape[1])
    channel_defs = scene.channels[:channel_count]
    stats = _sample_channel_windows(data)
    dtype_min, dtype_max = _dtype_window_bounds(data.dtype)

    channels = []
    for c_idx in range(channel_count):
        meta = channel_defs[c_idx] if c_idx < len(channel_defs) else None
        start, end = stats[c_idx]
        channels.append(
            {
                "label": _channel_label(meta, c_idx),
                "color": _channel_color(meta),
                "window": {
                    "start": _json_scalar(start),
                    "end": _json_scalar(end),
                    "min": _json_scalar(dtype_min),
                    "max": _json_scalar(dtype_max),
                },
                "active": True,
            }
        )
    return channels


def _sample_channel_windows(data) -> list[tuple[float, float]]:
    """Estimate display windows from a sparse per-channel sample."""
    sample = _sample_data(data)
    windows: list[tuple[float, float]] = []
    channel_count = int(sample.shape[1])
    for c_idx in range(channel_count):
        values = np.asarray(sample[:, c_idx, :, :, :]).ravel()
        if values.size == 0:
            windows.append((0.0, 0.0))
            continue
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            windows.append((0.0, 0.0))
            continue
        if np.issubdtype(finite.dtype, np.integer):
            start = float(np.percentile(finite, 0.1))
            end = float(np.percentile(finite, 99.9))
        else:
            start = float(np.percentile(finite, 1.0))
            end = float(np.percentile(finite, 99.0))
        if end < start:
            end = start
        windows.append((start, end))
    return windows


def _sample_data(data, target_values: int = 262144):
    """Subsample ``TCZYX`` data before computing display statistics.

    This is one of the few intentional eager materialization sites in the
    library. It is bounded by the strided sample size so OMERO window
    estimation does not compute the full scene into memory.
    """
    t, c, z, y, x = (int(v) for v in data.shape)
    total_values = max(1, t * c * z * y * x)
    stride = max(1, int(math.ceil((total_values / target_values) ** 0.25)))
    logger.debug("Sampling data for OMERO windows with stride=%d target_values=%d", stride, target_values)
    sampled = data[::stride, :, ::stride, ::stride, ::stride]
    return _materialize_sample(sampled)


def _materialize_sample(sampled):
    """Compute a bounded sample into NumPy for metadata-only statistics."""
    sample_values = getattr(sampled, "size", None)
    logger.debug("Materializing bounded sample for OMERO windows with size=%s", sample_values)
    return np.asarray(sampled.compute() if hasattr(sampled, "compute") else sampled)


def _dtype_window_bounds(dtype) -> tuple[float, float]:
    """Return the full numeric range of the array dtype for OME channel windows."""
    np_dtype = np.dtype(dtype)
    if np.issubdtype(np_dtype, np.integer):
        info = np.iinfo(np_dtype)
        return float(info.min), float(info.max)
    info = np.finfo(np_dtype)
    return float(info.min), float(info.max)


def _channel_label(meta, index: int) -> str:
    """Choose the label written into ``omero.channels``."""
    if meta is not None and meta.name:
        return meta.name
    return f"c{index}"


def _channel_color(meta) -> str:
    """Convert OME channel colors to six-digit RGB hex strings."""
    if meta is None or meta.color is None:
        return "FFFFFF"
    try:
        raw = int(meta.color)
    except Exception:
        return "FFFFFF"
    rgb = raw & 0xFFFFFF
    return f"{rgb:06X}"


def _json_scalar(value: float):
    """Convert NumPy-compatible scalar values into plain JSON numbers."""
    if float(value).is_integer():
        return int(value)
    return float(value)
