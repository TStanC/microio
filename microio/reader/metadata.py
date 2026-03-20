"""Metadata accessors and validated scene/level lookup helpers."""

from __future__ import annotations

from copy import deepcopy
import logging
from pathlib import Path

import dask.array as da
from microio.common.constants import AXES_ORDER
from microio.common.ngff import flattened_attrs
from microio.common.models import DataFlowReport, LevelRef, SceneRef, ValidationMessage
from microio.reader.ome_xml import OmeDocument, parse_ome_xml


logger = logging.getLogger("microio.reader.metadata")

def list_scene_refs(ds) -> list[SceneRef]:
    """Resolve all scenes in stable dataset order and cache the result.

    Parameters
    ----------
    ds:
        Open dataset handle.

    Returns
    -------
    list[SceneRef]
        Canonical scene references ordered by dataset order.
    """
    cached = ds._scene_refs_cache
    if cached is not None:
        logger.debug("Using cached scene refs for %s", ds.path)
        return cached

    logger.debug("Resolving scene refs for %s", ds.path)
    scene_ids = _ordered_scene_ids(ds)
    scenes: list[SceneRef] = []
    names: list[str] = []
    for index, scene_id in enumerate(scene_ids):
        name = _scene_name(ds, scene_id)
        names.append(name)
        scenes.append(
            SceneRef(
                id=scene_id,
                index=index,
                name=name,
                group_path=scene_id,
                ome_index=_resolve_ome_index(ds, scene_id, dataset_index=index, scene_name=name),
            )
        )

    counts: dict[str, int] = {}
    for name in names:
        counts[name] = counts.get(name, 0) + 1

    resolved = [
        SceneRef(
            id=scene.id,
            index=scene.index,
            name=scene.name,
            group_path=scene.group_path,
            ome_index=scene.ome_index,
            duplicate_name_count=counts[scene.name],
        )
        for scene in scenes
    ]
    ds._scene_refs_cache = resolved
    return resolved


def list_scenes(ds) -> list[str]:
    """List canonical scene ids in stable dataset order.

    Parameters
    ----------
    ds:
        Open dataset handle.

    Returns
    -------
    list[str]
        Scene ids in the same order returned by :func:`list_scene_refs`.
    """
    return [scene.id for scene in list_scene_refs(ds)]


def scene_ref(ds, scene: int | str) -> SceneRef:
    """Resolve a scene reference by id, dataset index, or unique name.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene:
        Scene selector. Integers are treated as dataset-order indexes. Strings
        are matched against scene ids first and then unique scene names.

        Examples: ``0`` resolves the first scene in dataset order, ``"0"``
        resolves the canonical scene id, and ``"C555"`` resolves a unique
        multiscale scene name.

    Returns
    -------
    SceneRef
        Canonical scene reference.

    Raises
    ------
    KeyError
        If the scene cannot be resolved or if a name is ambiguous.
    """
    refs = list_scene_refs(ds)
    if isinstance(scene, int):
        if 0 <= scene < len(refs):
            return refs[scene]
        raise KeyError(f"Scene index {scene} is out of range for dataset with {len(refs)} scenes")

    scene_text = str(scene)
    for ref in refs:
        if ref.id == scene_text:
            return ref

    name_matches = [ref for ref in refs if ref.name == scene_text]
    if len(name_matches) == 1:
        return name_matches[0]
    if len(name_matches) > 1:
        ids = [ref.id for ref in name_matches]
        raise KeyError(f"Scene name {scene_text!r} is ambiguous; matching ids={ids}")

    available = [ref.id for ref in refs]
    raise KeyError(f"Unknown scene reference {scene!r}; available scene ids={available}")


def classify_scene_reference(ds, value: int | str) -> str:
    """Classify a candidate scene reference without raising on misses.

    Returns
    -------
    str
        One of ``"index"``, ``"id"``, ``"name"``, ``"ambiguous_name"``, or
        ``"unknown"``.
    """
    if isinstance(value, int):
        return "index" if is_scene_index(ds, value) else "unknown"

    value_text = str(value)
    if is_scene_id(ds, value_text):
        return "id"
    matches = scene_name_matches(ds, value_text)
    if len(matches) == 1:
        return "name"
    if len(matches) > 1:
        return "ambiguous_name"
    return "unknown"


def is_scene_id(ds, value: str) -> bool:
    """Return whether ``value`` matches a canonical scene id."""
    value_text = str(value)
    return any(ref.id == value_text for ref in list_scene_refs(ds))


def is_scene_index(ds, value: int) -> bool:
    """Return whether ``value`` is a valid dataset-order scene index."""
    return any(ref.index == int(value) for ref in list_scene_refs(ds))


def scene_id_to_index(ds, scene_id: str) -> int:
    """Convert a canonical scene id into its dataset-order index."""
    return scene_ref(ds, str(scene_id)).index


def scene_index_to_id(ds, index: int) -> str:
    """Convert a dataset-order scene index into its canonical id."""
    return scene_ref(ds, int(index)).id


def scene_name_matches(ds, name: str) -> list[SceneRef]:
    """Return all scenes whose display name equals ``name``.

    Returns
    -------
    list[SceneRef]
        Matching scene references. The list may contain multiple items when
        names are duplicated in the dataset.
    """
    name_text = str(name)
    return [ref for ref in list_scene_refs(ds) if ref.name == name_text]


def root_metadata(ds) -> dict:
    """Return root-group attributes as a plain dictionary.

    Returns
    -------
    dict
        Root-group attributes with any Zarr v3 ``ome`` namespace projected to
        top-level semantic keys.
    """
    return flattened_attrs(ds.root)


def scene_metadata(ds, scene: int | str, *, corrected: bool = True) -> dict:
    """Read one scene's attrs, optionally overlaying stored repairs.

    Parameters
    ----------
    ds:
        Open dataset handle.
    scene:
        Scene selector accepted by :func:`scene_ref`.
    corrected:
        If ``True``, overlay persisted repair values onto the returned metadata.
        For example, a stored ``microio.repair.repaired_axes.z`` value is
        projected back onto the returned ``multiscales[0].datasets[*].scale``
        vectors.

    Returns
    -------
    dict
        Scene attributes as plain Python objects.
    """
    ref = scene_ref(ds, scene)
    attrs = _raw_scene_metadata(ds, ref.id)
    if not corrected:
        return deepcopy(attrs)
    logger.info("Applying repaired axes overlay to scene %s metadata", ref.id)
    return _apply_repaired_axes_overlay(attrs)


def multiscale_metadata(ds, scene: int | str, *, corrected: bool = True) -> dict:
    """Read and validate the primary multiscales block for one scene.

    The returned dictionary follows the semantic NGFF shape used throughout the
    library, for example::

        {
            "name": "scene",
            "axes": [
                {"name": "t", "type": "time"},
                {"name": "c", "type": "channel"},
                {"name": "z", "type": "space", "unit": "micrometer"},
                {"name": "y", "type": "space", "unit": "micrometer"},
                {"name": "x", "type": "space", "unit": "micrometer"},
            ],
            "datasets": [
                {
                    "path": "0",
                    "coordinateTransformations": [{"type": "scale", "scale": [1.0, 1.0, 0.5, 0.5, 0.5]}],
                }
            ],
        }

    Returns
    -------
    dict
        Validated multiscale metadata block for the scene.

    Raises
    ------
    ValueError
        If the scene has missing or unsupported axis metadata.
    """
    ref = scene_ref(ds, scene)
    attrs = scene_metadata(ds, ref.id, corrected=corrected)
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"Scene {ref.id} has no multiscales metadata")
    multiscale = multiscales[0]
    _validate_multiscale_axes(ref.id, multiscale)
    return multiscale


def list_levels(ds, scene: int | str) -> list[LevelRef]:
    """Enumerate and validate all multiscale levels for one scene.

    Returns
    -------
    list[LevelRef]
        Validated level descriptors ordered from finest to coarsest.

    Raises
    ------
    ValueError
        If the Zarr arrays and multiscale metadata are inconsistent.
    """
    ref = scene_ref(ds, scene)
    cached = ds._level_refs_cache.get(ref.id)
    if cached is not None:
        logger.debug("Using cached level refs for scene %s", ref.id)
        return cached

    logger.debug("Resolving level refs for scene %s", ref.id)
    scene_group = ds.root[ref.id]
    multiscale = multiscale_metadata(ds, ref.id)
    axes = tuple(axis["name"] for axis in multiscale["axes"])
    axis_units = tuple(axis.get("unit") for axis in multiscale["axes"])
    levels: list[LevelRef] = []
    previous_shape: tuple[int, ...] | None = None

    for level_index, dataset in enumerate(multiscale["datasets"]):
        path = str(dataset.get("path"))
        if path not in scene_group:
            raise ValueError(f"Scene {ref.id} level {path!r} is listed in multiscales but missing from the Zarr group")
        transforms = dataset.get("coordinateTransformations")
        if not isinstance(transforms, list) or not transforms:
            raise ValueError(f"Scene {ref.id} level {path!r} has no coordinate transformations")
        scale = transforms[0].get("scale")
        if not isinstance(scale, list):
            raise ValueError(f"Scene {ref.id} level {path!r} has malformed scale metadata")
        if len(scale) != len(axes):
            raise ValueError(
                f"Scene {ref.id} level {path!r} scale length {len(scale)} does not match axis count {len(axes)}"
            )
        array = scene_group[path]
        shape = tuple(int(dim) for dim in array.shape)
        if len(shape) != len(axes):
            raise ValueError(
                f"Scene {ref.id} level {path!r} ndim {len(shape)} does not match multiscale axis count {len(axes)}"
            )
        if previous_shape is not None:
            _validate_pyramid_shapes(ref.id, path, previous_shape, shape, axes)
        previous_shape = shape
        levels.append(
            LevelRef(
                scene_id=ref.id,
                level_index=level_index,
                path=path,
                shape=shape,
                dtype=str(array.dtype),
                scale=tuple(float(value) for value in scale),
                axis_names=axes,
                axis_units=axis_units,
            )
        )
    ds._level_refs_cache[ref.id] = levels
    return levels


def level_ref(ds, scene: int | str, level: int | str) -> LevelRef:
    """Resolve one level by index or path within a scene.

    Raises
    ------
    KeyError
        If the requested level index or path is unavailable.
    """
    ref = scene_ref(ds, scene)
    levels = list_levels(ds, ref.id)
    if isinstance(level, int):
        if 0 <= level < len(levels):
            return levels[level]
        raise KeyError(f"Scene {ref.id} level index {level} is out of range; available=0..{len(levels) - 1}")

    level_text = str(level)
    for candidate in levels:
        if candidate.path == level_text:
            return candidate
    raise KeyError(f"Scene {ref.id} has no level path {level_text!r}; available={[item.path for item in levels]}")


def read_level(ds, scene: int | str, level: int | str = 0):
    """Read one image level as a lazy Dask array.

    Returns
    -------
    dask.array.Array
        Lazy image data backed by the underlying Zarr array.
    """
    return _wrap_zarr_as_dask(read_level_zarr(ds, scene, level))


def read_level_zarr(ds, scene: int | str, level: int | str = 0):
    """Read one image level as the underlying Zarr array."""
    ref = scene_ref(ds, scene)
    level_info = level_ref(ds, ref.id, level)
    return ds.root[ref.id][level_info.path]


def read_level_numpy(ds, scene: int | str, level: int | str = 0):
    """Materialize one image level eagerly as a NumPy array.

    Returns
    -------
    numpy.ndarray
        Fully materialized array data for the requested level.
    """
    return read_level_zarr(ds, scene, level)[:]


def read_ome_xml(ds) -> str:
    """Read the dataset-level sidecar OME-XML text.

    Raises
    ------
    FileNotFoundError
        If ``OME/METADATA.ome.xml`` is missing.
    """
    xml_path = _ome_xml_path(ds.path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing OME sidecar XML: {xml_path}")
    return xml_path.read_text(encoding="utf-8", errors="replace")


def scene_ome_metadata(ds, scene: int | str):
    """Resolve normalized OME metadata for one scene.

    Returns
    -------
    SceneOmeMetadata
        Parsed scene metadata from the dataset sidecar OME-XML.

    Raises
    ------
    FileNotFoundError
        If the OME sidecar is missing.
    KeyError
        If the scene cannot be matched safely to an OME image entry.
    """
    ref = scene_ref(ds, scene)
    document = read_ome_document(ds)
    if ref.ome_index is not None and 0 <= ref.ome_index < len(document.scenes):
        candidate = document.scenes[ref.ome_index]
        if ref.name and candidate.name != ref.name and ref.duplicate_name_count == 1:
            logger.warning(
                "Scene %s matched XML by index but name differs: zarr=%r xml=%r",
                ref.id,
                ref.name,
                candidate.name,
            )
        return candidate

    unique_matches = [item for item in document.scenes if item.name == ref.name]
    if len(unique_matches) == 1:
        return unique_matches[0]
    raise KeyError(f"Could not match scene {ref.id!r} to OME-XML metadata")


def original_metadata(ds) -> dict[str, str]:
    """Return the OME ``OriginalMetadata`` key-value block."""
    return dict(read_ome_document(ds).original_metadata)


def validate_scene_data_flow(ds, scene: int | str) -> DataFlowReport:
    """Validate scene identity, multiscale access, and OME/Zarr consistency.

    Returns
    -------
    DataFlowReport
        Combined validation report for scene lookup, level access, and OME/Zarr
        consistency checks.
    """
    ref = scene_ref(ds, scene)
    logger.debug("Validating data flow for scene %s", ref.id)
    warnings: list[ValidationMessage] = []
    errors: list[ValidationMessage] = []
    try:
        levels = list_levels(ds, ref.id)
    except Exception as exc:
        errors.append(
            ValidationMessage(
                level="error",
                code="multiscale_invalid",
                message=f"Scene {ref.id} multiscale/array validation failed: {exc}",
            )
        )
        levels = []

    if ref.duplicate_name_count > 1:
        warnings.append(
            ValidationMessage(
                level="warning",
                code="duplicate_scene_name",
                message=f"Scene {ref.id} uses non-unique name {ref.name!r}; name-based reads must be disambiguated.",
            )
        )

    try:
        ome_scene = scene_ome_metadata(ds, ref.id)
    except FileNotFoundError:
        warnings.append(
            ValidationMessage(
                level="warning",
                code="ome_xml_missing",
                message=f"Scene {ref.id} has no OME sidecar XML; OME-backed validation is unavailable.",
            )
        )
    except KeyError as exc:
        errors.append(ValidationMessage(level="error", code="ome_scene_unmatched", message=str(exc)))
    else:
        if levels:
            level0 = levels[0]
            expected_shape = (
                int(ome_scene.size_t),
                int(ome_scene.size_c),
                int(ome_scene.size_z),
                int(ome_scene.size_y),
                int(ome_scene.size_x),
            )
            if level0.shape != expected_shape:
                errors.append(
                    ValidationMessage(
                        level="error",
                        code="ome_shape_mismatch",
                        message=f"Scene {ref.id} level 0 shape {level0.shape} does not match OME Pixels sizes {expected_shape}.",
                    )
                )
            if ref.name != ome_scene.name and ref.duplicate_name_count == 1:
                warnings.append(
                    ValidationMessage(
                        level="warning",
                        code="scene_name_mismatch",
                        message=f"Scene {ref.id} multiscale name {ref.name!r} differs from OME name {ome_scene.name!r}.",
                    )
                )

    return DataFlowReport(scene=ref, levels=levels, warnings=warnings, errors=errors)


def read_ome_document(ds) -> OmeDocument:
    """Read and cache the parsed OME document for the current dataset."""
    if ds._ome_document_cache is None:
        logger.debug("Parsing OME-XML for %s", ds.path)
        ds._ome_document_cache = parse_ome_xml(_ome_xml_path(ds.path).read_text(encoding="utf-8", errors="replace"))
    return ds._ome_document_cache


def _ordered_scene_ids(ds) -> list[str]:
    """Return scene ids in stable dataset order, preferring OME series metadata."""
    ome_group = ds.root.get("OME")
    if ome_group is not None:
        attrs = flattened_attrs(ome_group)
        series = attrs.get("series")
        if isinstance(series, list) and series:
            return [str(item) for item in series]

    names = [key for key, _ in ds.root.groups() if key != "OME"]
    return sorted(names, key=_natural_key)


def _scene_name(ds, scene_id: str) -> str:
    """Extract the display name for one scene from multiscales metadata."""
    attrs = flattened_attrs(ds.root[scene_id])
    multiscales = attrs.get("multiscales")
    if isinstance(multiscales, list) and multiscales:
        return str(multiscales[0].get("name") or scene_id)
    return str(scene_id)


def _resolve_ome_index(ds, scene_id: str, *, dataset_index: int, scene_name: str) -> int | None:
    """Map a scene to an OME image index using the safest available evidence."""
    try:
        document = read_ome_document(ds)
    except FileNotFoundError:
        return None

    if scene_id.isdigit():
        candidate = int(scene_id)
        if 0 <= candidate < len(document.scenes):
            return candidate

    if 0 <= dataset_index < len(document.scenes) and document.scenes[dataset_index].name == scene_name:
        return dataset_index

    matches = [item.index for item in document.scenes if item.name == scene_name]
    if len(matches) == 1:
        return matches[0]
    return None


def _validate_multiscale_axes(scene_id: str, multiscale: dict) -> None:
    """Validate axis metadata against the library's supported axis contract."""
    axes = multiscale.get("axes")
    datasets = multiscale.get("datasets")
    if not isinstance(axes, list) or not axes:
        raise ValueError(f"Scene {scene_id} has no multiscale axes metadata")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError(f"Scene {scene_id} has no multiscale datasets metadata")

    axis_names = [axis.get("name") for axis in axes]
    if any(not isinstance(name, str) or not name for name in axis_names):
        raise ValueError(f"Scene {scene_id} has unnamed multiscale axes")
    if len(set(axis_names)) != len(axis_names):
        raise ValueError(f"Scene {scene_id} has duplicate multiscale axes: {axis_names}")
    if tuple(axis_names) != AXES_ORDER:
        raise ValueError(f"Scene {scene_id} uses unsupported axis order {axis_names}; expected {list(AXES_ORDER)}")


def _validate_pyramid_shapes(
    scene_id: str,
    level_path: str,
    previous_shape: tuple[int, ...],
    current_shape: tuple[int, ...],
    axes: tuple[str, ...],
) -> None:
    """Ensure coarser pyramid levels do not grow relative to prior levels."""
    for axis_name, previous_dim, current_dim in zip(axes, previous_shape, current_shape):
        if current_dim <= 0:
            raise ValueError(f"Scene {scene_id} level {level_path!r} axis {axis_name} has invalid size {current_dim}")
        if current_dim > previous_dim:
            raise ValueError(
                f"Scene {scene_id} level {level_path!r} axis {axis_name} grows from {previous_dim} to {current_dim}"
            )


def _wrap_zarr_as_dask(array):
    """Wrap a Zarr-backed array as Dask while preserving existing chunking."""
    chunks = getattr(array, "chunks", None)
    if chunks is not None:
        return da.from_array(array, chunks=chunks, inline_array=True)
    return da.from_array(array, inline_array=True)


def _ome_xml_path(dataset_path: Path) -> Path:
    return Path(dataset_path) / "OME" / "METADATA.ome.xml"


def _raw_scene_metadata(ds, scene_id: str) -> dict:
    """Read and cache raw scene attrs without applying repair overlays."""
    cached = ds._raw_scene_metadata_cache.get(scene_id)
    if cached is None:
        logger.debug("Caching raw scene metadata for scene %s", scene_id)
        cached = flattened_attrs(ds.root[scene_id])
        ds._raw_scene_metadata_cache[scene_id] = cached
    return cached


def _apply_repaired_axes_overlay(attrs: dict) -> dict:
    """Overlay persisted repair values onto a scene attrs dictionary."""
    repaired_axes = attrs.get("microio", {}).get("repair", {}).get("repaired_axes", {})
    if not repaired_axes:
        return deepcopy(attrs)

    out = deepcopy(attrs)
    multiscales = out.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        return out
    multiscale = multiscales[0]
    axes = multiscale.get("axes")
    datasets = multiscale.get("datasets")
    if not isinstance(axes, list) or not isinstance(datasets, list):
        return out
    axis_index = {axis.get("name"): idx for idx, axis in enumerate(axes)}
    for axis_name, repair in repaired_axes.items():
        idx = axis_index.get(axis_name)
        if idx is None:
            continue
        unit = repair.get("unit")
        value = repair.get("value")
        if unit is not None:
            axes[idx]["unit"] = unit
        if value is None:
            continue
        for dataset in datasets:
            transforms = dataset.get("coordinateTransformations")
            if not isinstance(transforms, list) or not transforms:
                continue
            scale = transforms[0].get("scale")
            if not isinstance(scale, list) or len(scale) != len(AXES_ORDER):
                continue
            scale[idx] = float(value)
    return out


def _natural_key(text: str):
    return tuple(int(part) if part.isdigit() else part for part in _split_digits(text))


def _split_digits(text: str) -> list[str]:
    out: list[str] = []
    token = ""
    last_digit: bool | None = None
    for char in text:
        is_digit = char.isdigit()
        if last_digit is None or is_digit == last_digit:
            token += char
        else:
            out.append(token)
            token = char
        last_digit = is_digit
    if token:
        out.append(token)
    return out
