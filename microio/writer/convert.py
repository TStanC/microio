"""High-level conversion orchestration from proprietary sources to OME-Zarr."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import numpy as np

from microio.common.constants import DEFAULT_CHUNK_TARGET_MB, DEFAULT_TARGET_NGFF
from microio.common.models import ConversionReport, SceneReport
from .infer import infer_axis_resolution
from .ngff import create_root_store, write_root_ome_group, write_scene_image
from .scene_selection import select_scenes
from .source import BioformatsSource
from .tables import write_axes_trajectory_table
from .xmlparse import parse_ome_xml


logger = logging.getLogger("microio.writer.convert")


def convert_file(
    input_path: str | Path,
    output_path: str | Path,
    *,
    target_ngff: str = DEFAULT_TARGET_NGFF,
    include_scene_index: list[int] | None = None,
    include_scene_name: list[str] | None = None,
    exclude_scene_regex: list[str] | None = None,
    overwrite: bool = False,
    chunk_target_mb: int = DEFAULT_CHUNK_TARGET_MB,
    max_workers: int = 1,
    max_t: int | None = None,
    max_c: int | None = None,
    max_z: int | None = None,
) -> ConversionReport:
    """Convert one microscopy file into a multi-scene OME-Zarr dataset.

    Parameters
    ----------
    input_path:
        Path to the proprietary microscopy input file.
    output_path:
        Destination path for the output OME-Zarr directory.
    target_ngff:
        NGFF version string written into the output metadata.
    include_scene_index:
        Optional allow-list of zero-based scene indices.
    include_scene_name:
        Optional allow-list of scene names as reported by OME-XML.
    exclude_scene_regex:
        Optional list of regular expressions used to skip scenes by name.
    overwrite:
        Whether an existing output directory should be removed before writing.
    chunk_target_mb:
        Approximate chunk-size budget used by chunk estimation.
    max_workers:
        Reserved concurrency parameter. The current implementation remains
        serial and logs a warning when values above one are requested.
    max_t, max_c, max_z:
        Optional truncation of the written data along the time, channel, and
        axial dimensions. This is mainly useful for smoke tests and debugging.

    Returns
    -------
    ConversionReport
        Structured report containing per-scene status, warnings, and errors.
    """
    logger.info("Starting conversion: %s -> %s", input_path, output_path)

    input_path = Path(input_path)
    output_path = Path(output_path)
    source = BioformatsSource(input_path)

    ome_xml = source.get_ome_xml()
    _, scenes_xml, original_metadata = parse_ome_xml(ome_xml)
    bioio_scenes = source.get_scene_names()
    logger.info(
        "Scene discovery summary: total=%d metadata=%d",
        len(bioio_scenes),
        len(scenes_xml),
    )

    report = ConversionReport(input_path=input_path, output_path=output_path, target_ngff=target_ngff)
    _warn_runtime_limits(report, logger, max_workers=max_workers)

    selection = select_scenes(
        scenes_xml,
        bioio_scenes,
        include_scene_index,
        include_scene_name,
        exclude_scene_regex,
    )
    selected = selection.selected
    logger.info(
        "Scene selection summary: total=%d excluded=%d converting=%d",
        selection.total_count,
        selection.excluded_count,
        len(selected),
    )
    if not selected:
        msg = "No scenes selected after applying include/exclude filters"
        report.warnings.append(msg)
        logger.warning(msg)

    root = create_root_store(output_path, overwrite=overwrite)
    scene_ids: list[str] = []
    name_seen: dict[str, int] = {}
    total_selected = len(selected)
    for ordinal, scene in enumerate(selected, start=1):
        scene_id = _unique_scene_id(scene.name, name_seen)
        scene_ids.append(scene_id)
        scene_group = root.require_group(scene_id)
        srep = SceneReport(scene_index=scene.index, scene_id=scene_id, converted=False)
        logger.info("Converting scene %d/%d: %s (index=%d)", ordinal, total_selected, scene_id, scene.index)

        try:
            data = source.get_scene_dask(scene.index)
            logger.debug("Loaded scene %s with shape=%s dtype=%s", scene_id, getattr(data, "shape", None), getattr(data, "dtype", None))
            if max_t is not None:
                data = data[: max(1, int(max_t))]
                logger.debug("Applied max_t=%s to scene %s", max_t, scene_id)
            if max_c is not None:
                data = data[:, : max(1, int(max_c))]
                logger.debug("Applied max_c=%s to scene %s", max_c, scene_id)
            if max_z is not None:
                data = data[:, :, : max(1, int(max_z))]
                logger.debug("Applied max_z=%s to scene %s", max_z, scene_id)

            axis_res = infer_axis_resolution(scene, original_metadata)
            srep.axis_resolution = axis_res

            axis_scale = [axis_res["t"].value, 1.0, axis_res["z"].value, axis_res["y"].value, axis_res["x"].value]
            axis_units = [
                axis_res["t"].unit_normalized,
                None,
                axis_res["z"].unit_normalized,
                axis_res["y"].unit_normalized,
                axis_res["x"].unit_normalized,
            ]
            chunks = _estimate_chunks(tuple(data.shape), _itemsize_of(data.dtype), chunk_target_mb)
            logger.debug("Estimated chunks for scene %s: %s", scene_id, chunks)

            write_scene_image(
                scene_group,
                data,
                scene,
                axis_scale=axis_scale,
                axis_units=axis_units,
                ngff_version=target_ngff,
                chunks=chunks,
            )
            write_axes_trajectory_table(
                scene_group,
                scene,
                axis_res,
                size_t=int(data.shape[0]),
                size_c=int(data.shape[1]),
                size_z=int(data.shape[2]),
                original_metadata=original_metadata,
            )

            scene_group.attrs["microio"] = {
                "version": "0.1.0",
                "source_scene": {
                    "index": scene.index,
                    "name": scene.name,
                    "acquisition_date": scene.acquisition_date,
                    "instrument_ref": scene.instrument_ref,
                    "objective_settings_id": scene.objective_settings_id,
                },
                "axis_resolution": {
                    k: {
                        "value": v.value,
                        "unit_normalized": v.unit_normalized,
                        "unit_raw": v.unit_raw,
                        "source": v.source,
                        "confidence": v.confidence,
                        "fallback": v.fallback,
                        "warning_code": v.warning_code,
                    }
                    for k, v in axis_res.items()
                },
            }
            scene_group.attrs["bioformats2raw.layout"] = 3
            for axis_name, axis_info in axis_res.items():
                if axis_info.fallback:
                    logger.warning(
                        "Scene %s axis %s used fallback resolution value=%s unit=%s",
                        scene_id,
                        axis_name,
                        axis_info.value,
                        axis_info.unit_normalized,
                    )
                    report.fallback_events.append(
                        {
                            "scene_id": scene_id,
                            "axis": axis_name,
                            "warning_code": axis_info.warning_code or "fallback",
                        }
                    )
            srep.converted = True
            logger.info("Converted scene %s (index=%d)", scene_id, scene.index)
        except Exception as exc:
            logger.exception("Failed scene %s (index=%d)", scene_id, scene.index)
            srep.warnings.append(str(exc))
            report.errors.append(f"scene={scene_id}: {exc}")

        report.scene_reports.append(srep)

    write_root_ome_group(
        root,
        scenes=scene_ids,
        source_name=input_path.name,
        source_location=str(input_path.parent),
        ome_xml=ome_xml,
        acquisition_software=None,
        ngff_version=target_ngff,
    )

    logger.info("Conversion finished with %d scenes (%d errors)", len(report.scene_reports), len(report.errors))
    return report


def convert_many(inputs: list[str | Path], output_dir: str | Path, **kwargs) -> list[ConversionReport]:
    """Convert multiple proprietary inputs into sibling OME-Zarr datasets.

    Parameters
    ----------
    inputs:
        Source file paths to convert.
    output_dir:
        Destination directory where ``<stem>.zarr`` outputs will be created.
    **kwargs:
        Additional keyword arguments forwarded to :func:`convert_file`.

    Returns
    -------
    list[ConversionReport]
        One structured conversion report per input file.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: list[ConversionReport] = []
    for p in inputs:
        pth = Path(p)
        out = output_dir / f"{pth.stem}.zarr"
        logger.info("Queueing batch conversion for %s -> %s", pth, out)
        reports.append(convert_file(pth, out, **kwargs))
    return reports


def _unique_scene_id(name: str, seen: dict[str, int]) -> str:
    """Return a filesystem-safe, deterministic, collision-free scene id.

    Parameters
    ----------
    name:
        Raw scene name from OME-XML.
    seen:
        Mutable counter of identifiers already emitted in the current dataset.

    Returns
    -------
    str
        Sanitized scene identifier. Duplicate names receive ``__<n>`` suffixes.
    """
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._-") or "scene"
    idx = seen.get(cleaned, 0)
    seen[cleaned] = idx + 1
    return cleaned if idx == 0 else f"{cleaned}__{idx}"


def _warn_runtime_limits(report: ConversionReport, logger: logging.Logger, *, max_workers: int) -> None:
    """Emit warnings for unsupported or risky runtime settings.

    Parameters
    ----------
    report:
        Conversion report that collects user-visible warnings.
    logger:
        Logger used for runtime diagnostics.
    max_workers:
        Requested worker count from the caller.
    """
    if max_workers > 1:
        msg = "max_workers > 1 requested, but conversion currently runs serially"
        report.warnings.append(msg)
        logger.warning(msg)

    try:
        import psutil

        total_gib = psutil.virtual_memory().total / (1024 ** 3)
        if total_gib <= 10:
            msg = f"Low-memory host detected (~{total_gib:.1f} GiB RAM); consider smaller chunk_target_mb"
            report.warnings.append(msg)
            logger.warning(msg)
    except Exception:
        logger.debug("psutil unavailable; skipping host-memory advisory")


def _itemsize_of(dtype) -> int:
    """Resolve an array ``dtype`` itemsize for NumPy and NumPy-like wrappers.

    Parameters
    ----------
    dtype:
        Data type object or dtype-like token.

    Returns
    -------
    int
        Number of bytes per element.
    """
    try:
        return int(dtype.itemsize)
    except Exception:
        return int(np.dtype(dtype).itemsize)


def _estimate_chunks(shape: tuple[int, ...], itemsize: int, chunk_target_mb: int) -> tuple[int, int, int, int, int]:
    """Estimate a ``TCZYX`` chunk shape near the requested memory budget.

    Parameters
    ----------
    shape:
        Full array shape, expected in ``TCZYX`` order.
    itemsize:
        Element size in bytes.
    chunk_target_mb:
        Approximate chunk budget in mebibytes.

    Returns
    -------
    tuple[int, int, int, int, int]
        Chunk sizes in ``TCZYX`` order.
    """
    if len(shape) != 5:
        logger.warning("Chunk estimation expected 5D TCZYX data but received shape=%s", shape)
        return tuple(max(1, int(s)) for s in shape)  # type: ignore[return-value]

    t, c, z, y, x = (max(1, int(v)) for v in shape)
    budget_bytes = max(1, int(chunk_target_mb)) * 1024 * 1024

    # Favor c=1 chunks for interactive compatibility and lower random access cost.
    chunk_c = 1
    chunk_t = min(t, 1)
    chunk_z = min(z, max(1, int(math.sqrt(max(1, budget_bytes // max(itemsize, 1))))))

    # Remaining budget is used for 2D tile size.
    denom = max(1, itemsize * chunk_t * chunk_c * chunk_z)
    yx_budget = max(1, budget_bytes // denom)
    chunk_y = min(y, max(1, int(math.sqrt(yx_budget))))
    chunk_x = min(x, max(1, yx_budget // max(chunk_y, 1)))

    return (chunk_t, chunk_c, chunk_z, chunk_y, chunk_x)
