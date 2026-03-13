"""Metadata accessors and OME-XML-backed scene lookup helpers."""

from __future__ import annotations

from functools import lru_cache
import logging
from pathlib import Path

from microio.reader.ome_xml import OmeDocument, parse_ome_xml


logger = logging.getLogger("microio.reader.metadata")


def list_scenes(ds) -> list[str]:
    """List scene ids in stable dataset order."""
    ome_group = ds.root.get("OME")
    if ome_group is not None:
        attrs = ome_group.attrs.asdict()
        series = attrs.get("series")
        if isinstance(series, list) and series:
            return [str(item) for item in series]

    names = [key for key, _ in ds.root.groups() if key != "OME"]
    return sorted(names, key=_natural_key)


def root_metadata(ds) -> dict:
    return ds.root.attrs.asdict()


def scene_metadata(ds, scene_id: str, *, corrected: bool = True) -> dict:
    attrs = ds.root[scene_id].attrs.asdict()
    if corrected:
        return attrs
    return attrs


def multiscale_metadata(ds, scene_id: str) -> dict:
    attrs = ds.root[scene_id].attrs.asdict()
    multiscales = attrs.get("multiscales")
    if not isinstance(multiscales, list) or not multiscales:
        raise ValueError(f"Scene {scene_id} has no multiscales metadata")
    return multiscales[0]


def read_ome_xml(ds) -> str:
    xml_path = _ome_xml_path(ds.path)
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing OME sidecar XML: {xml_path}")
    return xml_path.read_text(encoding="utf-8", errors="replace")


def scene_ome_metadata(ds, scene_id: str):
    document = read_ome_document(ds.path)
    scene = _match_scene_to_xml(ds, scene_id, document)
    return scene


def original_metadata(ds) -> dict[str, str]:
    return read_ome_document(ds.path).original_metadata


@lru_cache(maxsize=16)
def read_ome_document(dataset_path: Path) -> OmeDocument:
    logger.debug("Parsing OME-XML for %s", dataset_path)
    return parse_ome_xml(_ome_xml_path(dataset_path).read_text(encoding="utf-8", errors="replace"))


def _ome_xml_path(dataset_path: Path) -> Path:
    return Path(dataset_path) / "OME" / "METADATA.ome.xml"


def _match_scene_to_xml(ds, scene_id: str, document: OmeDocument):
    multiscale_name = multiscale_metadata(ds, scene_id).get("name")
    if str(scene_id).isdigit():
        idx = int(scene_id)
        if 0 <= idx < len(document.scenes):
            candidate = document.scenes[idx]
            if multiscale_name and candidate.name != multiscale_name:
                logger.warning(
                    "Scene %s matched XML by index but name differs: zarr=%r xml=%r",
                    scene_id,
                    multiscale_name,
                    candidate.name,
                )
            return candidate

    if multiscale_name:
        for scene in document.scenes:
            if scene.name == multiscale_name:
                return scene
    raise KeyError(f"Could not match scene {scene_id!r} to OME-XML metadata")


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
