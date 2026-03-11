"""Bio-Formats source adapter used by the writer pipeline."""

from __future__ import annotations

import logging
from pathlib import Path

logger = logging.getLogger("microio.writer.source")


class BioformatsSource:
    """Thin adapter around ``bioio_bioformats.BioFile`` for read operations.

    Parameters
    ----------
    path:
        Path to the proprietary microscopy file handled by Bio-Formats.
    memoize:
        Bio-Formats memoization level passed through to ``BioFile``.
    original_meta:
        Whether vendor ``OriginalMetadata`` should be requested when reading
        metadata-heavy contexts such as OME-XML extraction.
    """

    def __init__(self, path: Path, memoize: int = 0, original_meta: bool = True):
        self.path = Path(path)
        self.memoize = memoize
        self.original_meta = original_meta
        logger.debug(
            "Initialized BioformatsSource(path=%s, memoize=%d, original_meta=%s)",
            self.path,
            self.memoize,
            self.original_meta,
        )

    def get_ome_xml(self) -> str:
        """Read the raw OME-XML block from the source file."""
        logger.debug("Opening metadata context for OME-XML extraction: %s", self.path)
        BioFile = _load_biofile()
        with BioFile(self.path, series=0, meta=True, original_meta=self.original_meta, memoize=self.memoize) as bf:
            logger.debug("Read OME-XML for %s", self.path)
            return bf.ome_xml

    def get_series_count(self) -> int:
        """Return the number of series exposed by Bio-Formats."""
        logger.debug("Querying Bio-Formats series count for %s", self.path)
        BioFile = _load_biofile()
        with BioFile(self.path, series=0, meta=False, original_meta=False, memoize=self.memoize) as bf:
            count = int(bf.core_meta.series_count)
            logger.info("Bio-Formats reported %d series for %s", count, self.path)
            return count

    def get_scene_names(self) -> list[str]:
        """Return BioIO scene names in their native order.

        The concrete BioIO/Bio-Formats wrapper API differs across releases, so
        this method probes a small set of stable-looking access patterns and
        normalizes the result to a list of strings.
        """
        logger.debug("Querying BioIO scenes for %s", self.path)
        BioFile = _load_biofile()
        with BioFile(self.path, series=0, meta=True, original_meta=False, memoize=self.memoize) as bf:
            scenes = self._extract_scene_names(bf)
            logger.info("BioIO reported %d scenes for %s", len(scenes), self.path)
            return scenes

    def get_scene_dask(self, series: int):
        """Open one scene as a lazy Dask array in BioFile's canonical order."""
        logger.debug("Opening scene %d from %s as Dask array", series, self.path)
        BioFile = _load_biofile()
        with BioFile(self.path, series=series, meta=False, original_meta=False, memoize=self.memoize) as bf:
            data = bf.to_dask(series=series)
            logger.debug(
                "Opened scene %d with shape=%s dtype=%s",
                series,
                getattr(data, "shape", None),
                getattr(data, "dtype", None),
            )
            return data

    @staticmethod
    def _extract_scene_names(bf) -> list[str]:
        """Resolve scene names from a BioFile instance across API variants."""
        candidates = []
        for attr_name in ("scenes", "scene_names"):
            if not hasattr(bf, attr_name):
                continue
            value = getattr(bf, attr_name)
            if callable(value):
                value = value()
            candidates = value
            if candidates:
                break

        if not candidates and hasattr(bf, "reader"):
            reader = bf.reader
            for attr_name in ("scenes", "scene_names"):
                if not hasattr(reader, attr_name):
                    continue
                value = getattr(reader, attr_name)
                if callable(value):
                    value = value()
                candidates = value
                if candidates:
                    break

        if not candidates:
            core_meta = getattr(bf, "core_meta", None)
            series_count = getattr(core_meta, "series_count", None)
            if series_count is None:
                raise RuntimeError(f"BioIO did not expose scene names for {getattr(bf, 'path', '<unknown>')}")
            return [f"Image:{idx}" for idx in range(int(series_count))]

        out: list[str] = []
        for idx, item in enumerate(candidates):
            if isinstance(item, str):
                out.append(item)
                continue
            scene_name = getattr(item, "name", None)
            if scene_name is not None:
                out.append(str(scene_name))
                continue
            out.append(str(item) if item is not None else f"Image:{idx}")
        return out


def _load_biofile():
    """Import BioFile lazily so unit tests can run without writer extras."""
    try:
        from bioio_bioformats.biofile import BioFile
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "bioio-bioformats is required for conversion. Install the package dependencies for the writer stack."
        ) from exc
    return BioFile
