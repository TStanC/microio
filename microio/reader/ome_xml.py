"""OME-XML parsing utilities for already-converted OME-Zarr datasets."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
import xml.etree.ElementTree as ET

from microio.common.models import SceneOmeMetadata


OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
NS = {"ome": OME_NS}


@dataclass(frozen=True)
class OmeDocument:
    """Parsed OME-XML document and lookup tables."""

    xml_text: str
    scenes: tuple[SceneOmeMetadata, ...]
    original_metadata: MappingProxyType[str, str]


def parse_ome_xml(xml_text: str) -> OmeDocument:
    """Parse OME-XML from the dataset sidecar."""
    root = ET.fromstring(xml_text)
    scenes: list[SceneOmeMetadata] = []
    for index, image in enumerate(root.findall("ome:Image", NS)):
        px = image.find("ome:Pixels", NS)
        if px is None:
            continue
        planes = []
        for plane in px.findall("ome:Plane", NS):
            planes.append(
                MappingProxyType(
                    {
                        "TheT": plane.get("TheT"),
                        "TheC": plane.get("TheC"),
                        "TheZ": plane.get("TheZ"),
                        "DeltaT": plane.get("DeltaT"),
                        "DeltaTUnit": plane.get("DeltaTUnit"),
                        "PositionX": plane.get("PositionX"),
                        "PositionXUnit": plane.get("PositionXUnit"),
                        "PositionY": plane.get("PositionY"),
                        "PositionYUnit": plane.get("PositionYUnit"),
                        "PositionZ": plane.get("PositionZ"),
                        "PositionZUnit": plane.get("PositionZUnit"),
                    }
                )
            )
        scenes.append(
            SceneOmeMetadata(
                index=index,
                name=image.get("Name") or str(index),
                size_t=int(px.get("SizeT", "1")),
                size_c=int(px.get("SizeC", "1")),
                size_z=int(px.get("SizeZ", "1")),
                size_y=int(px.get("SizeY", "1")),
                size_x=int(px.get("SizeX", "1")),
                physical_size_x=_maybe_float(px.get("PhysicalSizeX")),
                physical_size_x_unit=px.get("PhysicalSizeXUnit"),
                physical_size_y=_maybe_float(px.get("PhysicalSizeY")),
                physical_size_y_unit=px.get("PhysicalSizeYUnit"),
                physical_size_z=_maybe_float(px.get("PhysicalSizeZ")),
                physical_size_z_unit=px.get("PhysicalSizeZUnit"),
                time_increment=_maybe_float(px.get("TimeIncrement")),
                time_increment_unit=px.get("TimeIncrementUnit"),
                planes=tuple(planes),
            )
        )

    original_metadata: dict[str, str] = {}
    for item in root.findall(".//ome:OriginalMetadata", NS):
        key = item.find("ome:Key", NS)
        value = item.find("ome:Value", NS)
        if key is not None and value is not None and key.text is not None:
            original_metadata[key.text] = value.text or ""

    return OmeDocument(
        xml_text=xml_text,
        scenes=tuple(scenes),
        original_metadata=MappingProxyType(original_metadata),
    )


def _maybe_float(raw: str | None) -> float | None:
    """Parse a float-valued XML attribute, returning ``None`` on invalid input."""
    if raw is None:
        return None
    try:
        return float(raw)
    except Exception:
        return None
