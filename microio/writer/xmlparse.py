"""OME-XML parsing utilities based on ``xml.etree.ElementTree``."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import xml.etree.ElementTree as ET


OME_NS = "http://www.openmicroscopy.org/Schemas/OME/2016-06"
NS = {"ome": OME_NS}
logger = logging.getLogger("microio.writer.xmlparse")


@dataclass
class ChannelXmlMeta:
    """Channel metadata extracted from an OME ``Channel`` element."""

    index: int
    channel_id: str | None
    name: str | None
    color: str | None
    pinhole_size: float | None
    pinhole_size_unit: str | None


@dataclass
class SceneXmlMeta:
    """Normalized scene metadata extracted from OME-XML."""

    index: int
    name: str
    size_t: int
    size_c: int
    size_z: int
    size_y: int
    size_x: int
    dtype: str
    dimension_order: str
    physical_size_x: float | None
    physical_size_x_unit: str | None
    physical_size_y: float | None
    physical_size_y_unit: str | None
    physical_size_z: float | None
    physical_size_z_unit: str | None
    time_increment: float | None
    time_increment_unit: str | None
    acquisition_date: str | None
    instrument_ref: str | None
    objective_settings_id: str | None
    channels: list[ChannelXmlMeta]
    planes: list[dict[str, str | None]]


def parse_ome_xml(xml_text: str) -> tuple[ET.Element, list[SceneXmlMeta], dict[str, str]]:
    """Parse raw OME-XML into normalized scene and vendor metadata objects.

    Parameters
    ----------
    xml_text:
        Raw OME-XML string obtained from Bio-Formats.

    Returns
    -------
    tuple[xml.etree.ElementTree.Element, list[SceneXmlMeta], dict[str, str]]
        Parsed XML root element, normalized per-scene metadata records, and the
        flattened ``OriginalMetadata`` key/value map.
    """
    root = ET.fromstring(xml_text)
    images = root.findall(".//ome:Image", NS)
    logger.debug("Parsing OME-XML containing %d Image elements", len(images))

    scenes: list[SceneXmlMeta] = []
    for idx, image in enumerate(images):
        name = image.get("Name") or f"Image:{idx}"
        px = image.find("ome:Pixels", NS)
        if px is None:
            logger.warning("Skipping Image element %s because it has no Pixels child", name)
            continue
        channels = []
        for c_idx, channel in enumerate(px.findall("ome:Channel", NS)):
            channels.append(
                ChannelXmlMeta(
                    index=c_idx,
                    channel_id=channel.get("ID"),
                    name=channel.get("Name"),
                    color=channel.get("Color"),
                    pinhole_size=_f(channel.get("PinholeSize")),
                    pinhole_size_unit=channel.get("PinholeSizeUnit"),
                )
            )
        planes = []
        for plane in px.findall("ome:Plane", NS):
            planes.append(
                {
                    "TheT": plane.get("TheT"),
                    "TheC": plane.get("TheC"),
                    "TheZ": plane.get("TheZ"),
                    "DeltaT": plane.get("DeltaT"),
                    "DeltaTUnit": plane.get("DeltaTUnit"),
                    "ExposureTime": plane.get("ExposureTime"),
                    "ExposureTimeUnit": plane.get("ExposureTimeUnit"),
                    "PositionX": plane.get("PositionX"),
                    "PositionXUnit": plane.get("PositionXUnit"),
                    "PositionY": plane.get("PositionY"),
                    "PositionYUnit": plane.get("PositionYUnit"),
                    "PositionZ": plane.get("PositionZ"),
                    "PositionZUnit": plane.get("PositionZUnit"),
                }
            )

        acquisition_date_elem = image.find("ome:AcquisitionDate", NS)
        instrument_ref = image.find("ome:InstrumentRef", NS)
        objective_settings = image.find("ome:ObjectiveSettings", NS)

        scenes.append(
            SceneXmlMeta(
                index=idx,
                name=name,
                size_t=int(px.get("SizeT", "1")),
                size_c=int(px.get("SizeC", "1")),
                size_z=int(px.get("SizeZ", "1")),
                size_y=int(px.get("SizeY", "1")),
                size_x=int(px.get("SizeX", "1")),
                dtype=px.get("Type", "uint16"),
                dimension_order=px.get("DimensionOrder", "XYCZT"),
                physical_size_x=_f(px.get("PhysicalSizeX")),
                physical_size_x_unit=px.get("PhysicalSizeXUnit"),
                physical_size_y=_f(px.get("PhysicalSizeY")),
                physical_size_y_unit=px.get("PhysicalSizeYUnit"),
                physical_size_z=_f(px.get("PhysicalSizeZ")),
                physical_size_z_unit=px.get("PhysicalSizeZUnit"),
                time_increment=_f(px.get("TimeIncrement")),
                time_increment_unit=px.get("TimeIncrementUnit"),
                acquisition_date=acquisition_date_elem.text if acquisition_date_elem is not None else None,
                instrument_ref=instrument_ref.get("ID") if instrument_ref is not None else None,
                objective_settings_id=objective_settings.get("ID") if objective_settings is not None else None,
                channels=channels,
                planes=planes,
            )
        )
        logger.debug(
            "Parsed scene index=%d name=%s size_t=%d size_c=%d size_z=%d size_y=%d size_x=%d",
            idx,
            name,
            int(px.get("SizeT", "1")),
            int(px.get("SizeC", "1")),
            int(px.get("SizeZ", "1")),
            int(px.get("SizeY", "1")),
            int(px.get("SizeX", "1")),
        )

    original_metadata = {}
    for om in root.findall(".//ome:OriginalMetadata", NS):
        key = om.find("ome:Key", NS)
        value = om.find("ome:Value", NS)
        if key is not None and value is not None and key.text is not None:
            original_metadata[key.text] = value.text or ""

    logger.info(
        "Parsed OME-XML into %d scenes and %d OriginalMetadata entries",
        len(scenes),
        len(original_metadata),
    )
    return root, scenes, original_metadata


def _f(v: str | None) -> float | None:
    """Parse an XML numeric attribute into ``float`` or ``None``."""
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None
