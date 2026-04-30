from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
from copy import deepcopy
import json
from pathlib import Path
import shutil
import sys
import time
from typing import Any

import dask
import dask.array as da
from numcodecs import Blosc
import numpy as np
import zarr
from zarr.codecs import BloscCodec, ZstdCodec

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microio.reader.labels import list_labels as list_scene_labels
from microio.reader.open import open_dataset
from microio.workspace import (
    _normalize_workspace_labels,
    _single_label_level_multiscale,
    _single_scene_level_multiscale,
    _write_workspace_ome_group,
    _write_workspace_root_metadata,
)
from microio.common.ngff import node_zarr_format, non_ome_attrs, ome_metadata, replace_ome_attrs
from microio.writer.common import replace_node_ome_metadata
from microio.writer.images import _image_label_version


DATA_OUT = REPO_ROOT.parents[0] / "data_out"
BENCHMARK_ROOT = DATA_OUT / "benchmarks"


@dataclass(frozen=True)
class Candidate:
    name: str
    kind: str
    zarr_format: int | None = None
    chunks: tuple[int, ...] | None = None
    shards: tuple[int, ...] | None = None
    threads: int | None = 8
    compression: str = "source"
    include_labels: bool = False
    lock_writes: bool = False


def _chunk_512(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (
        1,
        1,
        max(1, int(shape[2])),
        min(512, int(shape[3])),
        min(512, int(shape[4])),
    )


def _chunk_1024(shape: tuple[int, ...]) -> tuple[int, ...]:
    return (
        1,
        1,
        max(1, int(shape[2])),
        min(1024, int(shape[3])),
        min(1024, int(shape[4])),
    )


def _shards_for(shape: tuple[int, ...], chunks: tuple[int, ...]) -> tuple[int, ...]:
    return (
        1,
        1,
        min(int(shape[2]), max(int(chunks[2]), 4)),
        min(int(shape[3]), max(int(chunks[3]) * 4, int(chunks[3]))),
        min(int(shape[4]), max(int(chunks[4]) * 4, int(chunks[4]))),
    )


def candidate_matrix(shape: tuple[int, ...], *, include_labels: bool) -> list[Candidate]:
    chunk_512 = _chunk_512(shape)
    chunk_1024 = _chunk_1024(shape)
    out = [
        Candidate("baseline_current_default", kind="baseline", threads=None, include_labels=include_labels),
        Candidate("baseline_current_threads8", kind="baseline", chunks=chunk_512, threads=8, include_labels=include_labels),
        Candidate("v2_lz4_chunk512_threads8", kind="custom", zarr_format=2, chunks=chunk_512, threads=8, compression="blosc_lz4", include_labels=include_labels),
        Candidate("v2_lz4_chunk1024_threads8", kind="custom", zarr_format=2, chunks=chunk_1024, threads=8, compression="blosc_lz4", include_labels=include_labels),
        Candidate("v3_lz4_chunk512_threads8", kind="custom", zarr_format=3, chunks=chunk_512, threads=8, compression="blosc_lz4", include_labels=include_labels),
        Candidate("v3_zstd_chunk512_threads8", kind="custom", zarr_format=3, chunks=chunk_512, threads=8, compression="zstd_fast", include_labels=include_labels),
        Candidate("v3_sharded_lz4_chunk512_threads8", kind="custom", zarr_format=3, chunks=chunk_512, shards=_shards_for(shape, chunk_512), threads=8, compression="blosc_lz4", include_labels=include_labels),
        Candidate("v3_sharded_zstd_chunk512_threads8", kind="custom", zarr_format=3, chunks=chunk_512, shards=_shards_for(shape, chunk_512), threads=8, compression="zstd_fast", include_labels=include_labels),
        Candidate("v3_sharded_lz4_chunk512_threads8_locked", kind="custom", zarr_format=3, chunks=chunk_512, shards=_shards_for(shape, chunk_512), threads=8, compression="blosc_lz4", include_labels=include_labels, lock_writes=True),
        Candidate("v3_sharded_zstd_chunk512_threads8_locked", kind="custom", zarr_format=3, chunks=chunk_512, shards=_shards_for(shape, chunk_512), threads=8, compression="zstd_fast", include_labels=include_labels, lock_writes=True),
        Candidate("v3_sharded_zstd_chunk512_threads1_locked", kind="custom", zarr_format=3, chunks=chunk_512, shards=_shards_for(shape, chunk_512), threads=1, compression="zstd_fast", include_labels=include_labels, lock_writes=True),
    ]
    return out


def _count_store_objects(path: Path) -> tuple[int, int, int]:
    file_count = 0
    dir_count = 0
    total_bytes = 0
    for entry in path.rglob("*"):
        if entry.is_dir():
            dir_count += 1
            continue
        file_count += 1
        total_bytes += entry.stat().st_size
    return file_count, dir_count, total_bytes


def _compression_kwargs(candidate: Candidate) -> dict[str, Any]:
    if candidate.kind == "baseline":
        return {}
    if candidate.zarr_format == 2:
        if candidate.compression == "blosc_lz4":
            return {"compressor": Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE)}
        raise ValueError(f"Unsupported v2 compression setting: {candidate.compression}")
    if candidate.compression == "blosc_lz4":
        return {"compressors": (BloscCodec(cname="lz4", clevel=5),)}
    if candidate.compression == "zstd_fast":
        return {"compressors": (ZstdCodec(level=1),)}
    raise ValueError(f"Unsupported v3 compression setting: {candidate.compression}")


def _write_candidate_array(
    group,
    name: str,
    data,
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    threads: int,
    dimension_names: tuple[str, ...] | None,
    compression_kwargs: dict[str, Any],
    lock_writes: bool,
):
    target = group.create_array(
        name,
        shape=tuple(int(dim) for dim in data.shape),
        dtype=np.dtype(data.dtype),
        chunks=chunks,
        shards=shards,
        dimension_names=dimension_names,
        overwrite=True,
        write_data=False,
        **compression_kwargs,
    )
    with dask.config.set(scheduler="threads", num_workers=max(1, int(threads))):
        da.store(data.rechunk(chunks), target, lock=lock_writes, compute=True)
    return target


def _write_scene_candidate(
    ds,
    root,
    scene_id: str,
    source_level_path: str,
    source,
    *,
    chunks: tuple[int, ...],
    shards: tuple[int, ...] | None,
    threads: int,
    compression_kwargs: dict[str, Any],
    lock_writes: bool,
) -> None:
    scene_group = root.create_group(scene_id)
    attrs = ds.read_scene_metadata(scene_id)
    attrs["multiscales"] = [_single_scene_level_multiscale(ds, scene_id, source_level_path)]
    scene_ome = ome_metadata(ds.root[scene_id])
    scene_ome["multiscales"] = [deepcopy(attrs["multiscales"][0])]
    if "omero" in attrs:
        scene_ome["omero"] = deepcopy(attrs["omero"])
    replace_ome_attrs(scene_group, scene_ome, extra_attrs=non_ome_attrs(ds.root[scene_id]))
    _write_candidate_array(
        scene_group,
        "0",
        source,
        chunks=chunks,
        shards=shards,
        threads=threads,
        dimension_names=tuple(axis["name"] for axis in attrs["multiscales"][0]["axes"]) if node_zarr_format(scene_group) >= 3 else None,
        compression_kwargs=compression_kwargs,
        lock_writes=lock_writes,
    )


def _carry_labels_candidate(
    ds,
    root,
    scene_id: str,
    source_level_path: str,
    labels: list[str],
    *,
    chunks: tuple[int, ...],
    threads: int,
    compression_kwargs: dict[str, Any],
    lock_writes: bool,
) -> None:
    if not labels:
        return
    scene_group = root[scene_id]
    labels_group = scene_group.create_group("labels")
    replace_node_ome_metadata(labels_group, {"labels": sorted(labels)})
    for name in labels:
        source_group = ds.root[scene_id]["labels"][name]
        label_level = ds.label_level_ref(scene_id, name, source_level_path)
        label_data = ds.read_label(scene_id, name, label_level.path)
        target_group = labels_group.create_group(name)
        target_chunks = tuple(int(max(1, min(dim, chunk))) for dim, chunk in zip(label_level.shape, chunks, strict=True))
        _write_candidate_array(
            target_group,
            "0",
            label_data,
            chunks=target_chunks,
            shards=None,
            threads=threads,
            dimension_names=label_level.axis_names if node_zarr_format(target_group) >= 3 else None,
            compression_kwargs=compression_kwargs,
            lock_writes=lock_writes,
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


def _run_baseline(ds, scene_id: str, destination: Path, candidate: Candidate, labels: list[str]) -> None:
    kwargs: dict[str, Any] = {"overwrite": True, "threads": candidate.threads}
    if candidate.chunks is not None:
        kwargs["chunks"] = candidate.chunks
    if labels:
        kwargs["labels"] = labels
    ds.create_workspace(destination, scene_id, **kwargs)


def _run_custom(ds, scene_id: str, destination: Path, candidate: Candidate, labels: list[str]) -> None:
    if destination.exists():
        shutil.rmtree(destination, ignore_errors=True)
    level = ds.level_ref(scene_id, 0)
    source = ds.read_level(scene_id, level.path)
    root = zarr.open(destination, mode="w", zarr_format=int(candidate.zarr_format))
    _write_workspace_root_metadata(ds, root, scene_id, level, chunks=candidate.chunks, labels=labels)
    _write_workspace_ome_group(ds, root, destination, scene_id)
    compression_kwargs = _compression_kwargs(candidate)
    _write_scene_candidate(
        ds,
        root,
        scene_id,
        level.path,
        source,
        chunks=tuple(int(chunk) for chunk in candidate.chunks or _chunk_512(level.shape)),
        shards=candidate.shards,
        threads=candidate.threads,
        compression_kwargs=compression_kwargs,
        lock_writes=candidate.lock_writes,
    )
    _carry_labels_candidate(
        ds,
        root,
        scene_id,
        level.path,
        labels,
        chunks=tuple(int(chunk) for chunk in candidate.chunks or _chunk_512(level.shape)),
        threads=candidate.threads,
        compression_kwargs=compression_kwargs,
        lock_writes=candidate.lock_writes,
    )


def benchmark_dataset(
    dataset: Path,
    scene: str,
    *,
    repeats: int,
    output_root: Path,
    label_name: str | None,
    candidate_substrings: list[str] | None = None,
) -> list[dict[str, Any]]:
    ds = open_dataset(dataset, mode="a")
    source_shape = tuple(int(dim) for dim in ds.level_ref(scene, 0).shape)
    requested_labels = _normalize_workspace_labels(ds, scene, [label_name]) if label_name else []
    candidates = candidate_matrix(source_shape, include_labels=bool(requested_labels))
    if candidate_substrings:
        lowered = [item.lower() for item in candidate_substrings]
        candidates = [candidate for candidate in candidates if any(token in candidate.name.lower() for token in lowered)]
    output_root.mkdir(parents=True, exist_ok=True)
    results: list[dict[str, Any]] = []
    for candidate in candidates:
        for repeat in range(repeats):
            destination = output_root / f"{dataset.stem}_{scene}_{candidate.name}_run{repeat + 1}.zarr"
            shutil.rmtree(destination, ignore_errors=True)
            started = time.perf_counter()
            error = None
            try:
                if candidate.kind == "baseline":
                    _run_baseline(ds, scene, destination, candidate, requested_labels)
                else:
                    _run_custom(ds, scene, destination, candidate, requested_labels)
                elapsed = time.perf_counter() - started
                workspace_ds = open_dataset(destination)
                flow = workspace_ds.validate_scene_data_flow(scene)
                files, dirs, total_bytes = _count_store_objects(destination)
                results.append(
                    {
                        "dataset": str(dataset),
                        "scene": str(scene),
                        "candidate": candidate.name,
                        "kind": candidate.kind,
                        "repeat": repeat + 1,
                        "elapsed_seconds": elapsed,
                        "files": files,
                        "dirs": dirs,
                        "bytes": total_bytes,
                        "throughput_mib_s": (total_bytes / (1024 * 1024)) / elapsed if elapsed > 0 else None,
                        "flow_errors": [asdict(item) for item in flow.errors],
                        "flow_warnings": [asdict(item) for item in flow.warnings],
                        "chunks": list(candidate.chunks) if candidate.chunks is not None else None,
                        "shards": list(candidate.shards) if candidate.shards is not None else None,
                        "threads": candidate.threads,
                        "zarr_format": candidate.zarr_format,
                        "compression": candidate.compression,
                        "include_labels": bool(requested_labels),
                        "lock_writes": candidate.lock_writes,
                    }
                )
            except Exception as exc:
                elapsed = time.perf_counter() - started
                error = repr(exc)
                results.append(
                    {
                        "dataset": str(dataset),
                        "scene": str(scene),
                        "candidate": candidate.name,
                        "kind": candidate.kind,
                        "repeat": repeat + 1,
                        "elapsed_seconds": elapsed,
                        "error": error,
                        "chunks": list(candidate.chunks) if candidate.chunks is not None else None,
                        "shards": list(candidate.shards) if candidate.shards is not None else None,
                        "threads": candidate.threads,
                        "zarr_format": candidate.zarr_format,
                        "compression": candidate.compression,
                        "include_labels": bool(requested_labels),
                        "lock_writes": candidate.lock_writes,
                    }
                )
            finally:
                if error is None:
                    shutil.rmtree(destination, ignore_errors=True)
    return results


def default_jobs() -> list[tuple[Path, str, str | None]]:
    jobs = [
        (REPO_ROOT.parents[0] / "data_in" / "zarr" / "vsi_test.zarr", "0", None),
        (REPO_ROOT.parents[0] / "data_in" / "zarr" / "lif_test.zarr", "15", None),
    ]
    labeled_source = REPO_ROOT.parents[0] / "data_out" / "debug_fluct_70365df9dec64789acddf36f246020e1.zarr"
    if labeled_source.exists():
        label_name = list_scene_labels(open_dataset(labeled_source), "0")
        jobs.append((labeled_source, "0", label_name[0] if label_name else None))
    return jobs


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark workspace creation strategies for microio.")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeats per candidate. Default: 1.")
    parser.add_argument(
        "--output",
        type=Path,
        default=BENCHMARK_ROOT / "workspace_creation_report.json",
        help="Path to the JSON report. Default: data_out/benchmarks/workspace_creation_report.json",
    )
    parser.add_argument("--dataset", type=Path, default=None, help="Optional single dataset path to benchmark.")
    parser.add_argument("--scene", default="0", help="Scene id for --dataset. Default: 0.")
    parser.add_argument("--label-name", default=None, help="Optional carried label name for --dataset.")
    parser.add_argument(
        "--candidate-substring",
        action="append",
        default=None,
        help="Only run candidates whose names contain this substring. May be passed multiple times.",
    )
    args = parser.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    run_root = output_path.parent / "workspace_runs"
    shutil.rmtree(run_root, ignore_errors=True)
    run_root.mkdir(parents=True, exist_ok=True)

    jobs = [(Path(args.dataset), str(args.scene), args.label_name)] if args.dataset is not None else default_jobs()

    all_results: list[dict[str, Any]] = []
    for dataset, scene, label_name in jobs:
        print(f"Benchmarking {dataset} scene={scene} label={label_name}")
        all_results.extend(
            benchmark_dataset(
                dataset,
                scene,
                repeats=max(1, int(args.repeats)),
                output_root=run_root,
                label_name=label_name,
                candidate_substrings=args.candidate_substring,
            )
        )

    output_path.write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    print(f"Wrote {len(all_results)} benchmark records to {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
