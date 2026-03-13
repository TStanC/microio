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

Out of scope:
- proprietary LIF/VSI conversion
- automatic `x/y` repair
- automatic scalar `t` repair from ambiguous timing metadata

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

Reader safety rules:
- scene `id` means the actual Zarr child key, for example `"15"`
- scene `index` means the zero-based dataset order returned by `list_scenes()`
- scene names may be duplicated in real datasets; ambiguous name lookup raises an error
- high-level image reads are lazy and Dask-first by default
- level access validates metadata paths, scale vectors, axis order, and array dimensionality before returning data

*Disclaimer: OpenAI's codex extension was heavily used to program this library*
