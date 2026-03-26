# microio

`microio` is a reader-first microscopy I/O library for inspecting and enriching
existing bioformats2raw-style OME-Zarr datasets on disk.

It focuses on safe access to scene metadata and image pyramids, plus a small
set of write-side enrichments for data that already lives in a Zarr store.

## Installation

```bash
pip install git+https://github.com/TStanC/microio.git
```

## What It Does

- Opens existing OME-Zarr datasets through `open_dataset()`
- Resolves scenes by canonical id, dataset index, or unique scene name
- Reads root, scene, multiscale, and OME-XML metadata
- Validates multiscale axis metadata and level-to-array consistency
- Exposes scene and label levels as lazy Dask arrays, raw Zarr arrays, or eager NumPy arrays
- Builds and loads per-plane coordinate tables from OME plane metadata
- Repairs placeholder `z` metadata when stronger OME evidence exists
- Writes scene-local tables, NGFF label images, timepoint-scoped label updates, and single-scale ROI cutouts

## Safety Boundaries

- `microio` operates on datasets that already exist on disk
- It does not convert proprietary formats such as LIF or VSI into OME-Zarr
- It does not invent `x` or `y` calibration
- Scalar `t` repair is intentionally conservative and may remain unresolved
- Name-based scene lookup is only accepted when a name is unique within the dataset
- High-level image reads validate multiscale metadata before returning data

## Quick Start

```python
from microio import open_dataset

ds = open_dataset("path/to/dataset.zarr")

scene = ds.scene_ref(0)
print(scene.id, scene.name)

root_md = ds.read_root_metadata()
scene_md = ds.read_scene_metadata(scene.id)
ome_scene = ds.read_scene_ome_metadata(scene.id)
levels = ds.list_levels(scene.id)
labels = ds.list_labels(scene.id)

level1 = ds.read_level(scene.id, 1)
level1_zarr = ds.read_level_zarr(scene.id, 1)
level1_numpy = ds.read_level_numpy(scene.id, 1)

if labels:
    label = ds.get_label(scene.id, labels[0])
    label_md = label.metadata()
    label_level0 = label.array(0)
```

## Repair And Plane Tables

```python
from microio import open_dataset

ds = open_dataset("path/to/dataset.zarr", mode="a")

table, table_report = ds.ensure_plane_table("0", filetype="vsi")
repair_report = ds.repair_axis_metadata("0", persist=True, filetype="vsi")

print(table_report.row_count, table_report.persisted)
print(repair_report.axis_states["z"])
```

`ensure_plane_table()` reuses a compatible stored table when possible and
rebuilds it from OME plane metadata when needed. `repair_axis_metadata()`
persists only accepted repairs and leaves unresolved metadata unchanged.

## Limited Write-Side Enrichment

```python
import numpy as np

from microio import open_dataset

ds = open_dataset("path/to/dataset.zarr", mode="a")

ds.write_table("0", "measurements", {"label_id": [1, 2], "volume": [10.5, 12.0]})

ds.write_label_image(
    "0",
    "segmentation",
    np.zeros(ds.level_ref("0", 0).shape, dtype=np.uint16),
    colors=[
        {"label-value": 0, "rgba": [0, 0, 0, 0]},
        {"label-value": 1, "rgba": [0, 255, 0, 255]},
    ],
)

ds.write_label_timepoint(
    "0",
    "segmentation_by_t",
    np.zeros((1, 1, *ds.level_ref("0", 0).shape[2:]), dtype=np.uint16),
    timepoint=0,
    attrs={"source_channel": 0},
)

ds.write_roi(
    "0",
    "roi_1",
    {"t": (0, 2), "z": (0, 5), "y": (100, 300), "x": (200, 400)},
)
```

Notes:

- `write_table()` also accepts pandas `DataFrame` input when pandas is installed
- Write APIs are fail-safe by default and require `overwrite=True` to replace an existing target
- `write_label_image()` writes integer label pyramids aligned to the source image pyramid
- `write_label_timepoint()` initializes or reuses a label pyramid and writes one timepoint with caller-coordinated overwrite protection
- Label-image user attrs are stored under `label_group.attrs["microio"]["label-attrs"]`
- Label-image channel size may match the source image channel size or use `1` when one label volume applies to all source channels
- `write_roi()` is a microio extension for single-scale cutouts and is not stored as an NGFF label image

## CLI

Inspect a dataset:

```bash
microio inspect --input path/to/dataset.zarr --log-level INFO
```

Repair scene metadata and optionally persist plane tables:

```bash
microio repair --input path/to/dataset.zarr --scene 0 --filetype vsi --persist-table --persist --log-level DEBUG
```

## Main API Surface

- `open_dataset(path, mode="r")`
- `DatasetHandle.list_scene_refs()`, `list_scenes()`, `scene_ref()`
- `DatasetHandle.read_root_metadata()`, `read_scene_metadata()`, `read_multiscale_metadata()`
- `DatasetHandle.read_scene_ome_metadata()`, `read_original_metadata()`
- `DatasetHandle.list_levels()`, `level_ref()`, `read_level()`, `read_level_zarr()`, `read_level_numpy()`
- `DatasetHandle.list_labels()`, `get_label()`, `read_label_metadata()`, `list_label_levels()`
- `DatasetHandle.label_level_ref()`, `read_label()`, `read_label_zarr()`, `read_label_numpy()`
- `DatasetHandle.validate_scene_data_flow()`
- `DatasetHandle.build_plane_table()`, `ensure_plane_table()`, `load_table()`
- `DatasetHandle.inspect_axis_metadata()`, `repair_axis_metadata()`, `list_rois()`, `load_roi()`
- `DatasetHandle.write_table()`, `write_label_image()`, `write_label_timepoint()`, `write_roi()`

For implementation details, consult the package docstrings or DeepWiki.

This library was created with the help of OpenAIs CODEX agent.

[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/TStanC/microio)
