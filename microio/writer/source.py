"""Bio-Formats source adapter used by the writer pipeline."""

from __future__ import annotations

from pathlib import Path

from bioio_bioformats.biofile import BioFile


class BioformatsSource:
    """Adapter around ``bioio_bioformats.BioFile`` for read operations."""

    def __init__(self, path: Path, memoize: int = 0, original_meta: bool = True):
        self.path = Path(path)
        self.memoize = memoize
        self.original_meta = original_meta

    def get_ome_xml(self) -> str:
        """Read raw OME-XML once via series 0 metadata context."""
        with BioFile(self.path, series=0, meta=True, original_meta=self.original_meta, memoize=self.memoize) as bf:
            return bf.ome_xml

    def get_series_count(self) -> int:
        """Return number of available series/scenes."""
        with BioFile(self.path, series=0, meta=False, original_meta=False, memoize=self.memoize) as bf:
            return int(bf.core_meta.series_count)

    def get_scene_dask(self, series: int):
        """Return scene image as dask array in canonical BioFile output order."""
        with BioFile(self.path, series=series, meta=False, original_meta=False, memoize=self.memoize) as bf:
            return bf.to_dask(series=series)
