"""Dataset opening helpers for bioformats2raw OME-Zarr datasets."""

from __future__ import annotations

import logging
from pathlib import Path

import zarr

from microio.common.models import DatasetHandle
from microio.reader.ome_xml import parse_ome_xml


logger = logging.getLogger("microio.reader.open")


def open_dataset(path: str | Path, *, mode: str = "r", ome_scene_map: dict[str, int] | None = None) -> DatasetHandle:
    """Open an existing OME-Zarr dataset and return a dataset handle.

    Parameters
    ----------
    path:
        Filesystem path to an existing bioformats2raw-style OME-Zarr store.
    mode:
        Zarr access mode passed through to :func:`zarr.open`. Use ``"r"`` for
        read-only inspection and ``"a"`` when persisting repairs, tables, or
        other microio-managed enrichments.
    ome_scene_map:
        Optional explicit mapping from canonical Zarr scene ids to OME image
        indexes. Use this only when the dataset cannot be matched safely through
        unique names or validated dataset-order matching.

        Example: ``open_dataset("sample.zarr", mode="a")`` is required before
        calling :meth:`DatasetHandle.repair_axis_metadata` with
        ``persist=True``.

    Returns
    -------
    DatasetHandle
        Handle exposing validated scene lookup, metadata access, image reads,
        repair helpers, and limited write-side enrichment methods.

    Raises
    ------
    FileNotFoundError
        If the requested path does not exist.
    ValueError
        If Zarr rejects the supplied mode or cannot interpret the store.

    Examples
    --------
    >>> ds = open_dataset("plate.zarr")
    >>> ds.list_scenes()
    ['0']
    >>> writable = open_dataset("plate.zarr", mode="a")
    """
    dataset_path = Path(path)
    logger.info("Opening dataset %s with mode=%s", dataset_path, mode)
    if mode == "a" and not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path does not exist: {dataset_path}")
    root = zarr.open(dataset_path, mode=mode)
    validated_map = _validate_ome_scene_map(dataset_path, root, ome_scene_map)
    return DatasetHandle(path=dataset_path, root=root, mode=mode, ome_scene_map=validated_map)


def _validate_ome_scene_map(dataset_path: Path, root, ome_scene_map: dict[str, int] | None) -> dict[str, int] | None:
    if ome_scene_map is None:
        return None
    if not isinstance(ome_scene_map, dict):
        raise TypeError("ome_scene_map must be a mapping from scene id to OME image index")

    scene_ids = {str(key) for key, _ in root.groups() if str(key) != "OME"}
    normalized: dict[str, int] = {}
    used_indexes: set[int] = set()
    xml_path = Path(dataset_path) / "OME" / "METADATA.ome.xml"
    if not xml_path.exists():
        raise FileNotFoundError("ome_scene_map requires OME/METADATA.ome.xml so mapped indexes can be validated")
    document = parse_ome_xml(xml_path.read_text(encoding="utf-8", errors="replace"))
    scene_count = len(document.scenes)

    for raw_scene_id, raw_index in ome_scene_map.items():
        scene_id = str(raw_scene_id)
        if scene_id not in scene_ids:
            raise ValueError(f"ome_scene_map references unknown scene id {scene_id!r}")
        if isinstance(raw_index, bool) or not isinstance(raw_index, int):
            raise TypeError(f"ome_scene_map index for scene {scene_id!r} must be an integer")
        ome_index = int(raw_index)
        if not 0 <= ome_index < scene_count:
            raise ValueError(f"ome_scene_map index {ome_index} for scene {scene_id!r} is out of bounds for {scene_count} OME scenes")
        if ome_index in used_indexes:
            raise ValueError(f"ome_scene_map index {ome_index} is assigned more than once")
        normalized[scene_id] = ome_index
        used_indexes.add(ome_index)
    return normalized
