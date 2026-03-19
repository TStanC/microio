# microio library
## Inspect and enrich bioformats2raw OME-Zarr microscopy data

`microio` is now a reader-first library for already-converted OME-Zarr datasets.

Current scope:
- open bioformats2raw-style OME-Zarr datasets
- parse scene metadata from Zarr attrs and `OME/METADATA.ome.xml`
- resolve scenes safely by id, dataset index, or unique name
- inspect canonical scene mappings and multiscale level metadata
- read specific multiscale levels as lazy Dask arrays with validation against Zarr metadata
- validate multiscale axis metadata
- build per-plane coordinate tables
- safely repair placeholder `z` metadata when stronger XML evidence exists
- write scene-local tables, single-scale label images, and single-scale ROI cutouts programmatically

Out of scope:
- proprietary LIF/VSI conversion
- automatic `x/y` repair
- automatic scalar `t` repair from ambiguous timing metadata
- CLI-driven authoring of new write-side enrichments

Basic reader usage:

```python
from microio import open_dataset

ds = open_dataset("path/to/dataset.zarr")

scene = ds.scene_ref(0)  # dataset index -> canonical scene ref
levels = ds.list_levels(scene.id)
array = ds.read_level(scene.id, 1)  # validated lazy Dask access to one pyramid level
raw = ds.read_level_zarr(scene.id, 1)  # explicit raw Zarr access
eager = ds.read_level_numpy(scene.id, 1)  # explicit NumPy materialization

# Name lookup is allowed only when the name is unique.
matches = ds.scene_name_matches("C555")
```

Programmatic write-side enrichment:

```python
import numpy as np

from microio import open_dataset

ds = open_dataset("path/to/dataset.zarr", mode="a")
ds.write_table("0", "measurements", {"label_id": [1, 2], "volume": [10.5, 12.0]})
ds.write_label_image("0", "segmentation", np.zeros(ds.level_ref("0", 0).shape, dtype=np.uint16))
ds.write_roi("0", "roi_1", {"t": (0, 2), "z": (0, 5), "y": (100, 300), "x": (200, 400)})
```

Notes:
- pandas `DataFrame` input is supported for `write_table` when pandas is installed
- write APIs are fail-safe by default and require `overwrite=True` to replace existing targets
- the CLI remains limited to inspection and repair flows

Reader safety rules:
- scene `id` means the actual Zarr child key, for example `"15"`
- scene `index` means the zero-based dataset order returned by `list_scenes()`
- scene names may be duplicated in real datasets; ambiguous name lookup raises an error
- high-level image reads are lazy and Dask-first by default
- level access validates metadata paths, scale vectors, axis order, and array dimensionality before returning data

*For more detailed documentation, please visit [DeepWiki](https://deepwiki.com/TStanC/microio) or look at the docstrings.*

*Disclaimer: OpenAI's codex extension was heavily used to program this library*
