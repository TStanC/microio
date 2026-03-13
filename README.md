# microio library
## Inspect and enrich bioformats2raw OME-Zarr microscopy data

`microio` is now a reader-first library for already-converted OME-Zarr datasets.

Current scope:
- open bioformats2raw-style OME-Zarr datasets
- parse scene metadata from Zarr attrs and `OME/METADATA.ome.xml`
- validate multiscale axis metadata
- build per-plane coordinate tables
- safely repair placeholder `z` metadata when stronger XML evidence exists

Out of scope:
- proprietary LIF/VSI conversion
- automatic `x/y` repair
- automatic scalar `t` repair from ambiguous timing metadata

*Disclaimer: OpenAI's codex extension was heavily used to program this library*
