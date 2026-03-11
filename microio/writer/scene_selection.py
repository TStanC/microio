"""Scene selection helpers with BioIO/OME-XML reconciliation."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import re


logger = logging.getLogger("microio.writer.scene_selection")


@dataclass(frozen=True)
class SceneSelection:
    """Scene selection result with accounting for logging and validation."""

    selected: list
    total_count: int
    excluded_count: int


def select_scenes(scenes_xml, bioio_scenes, include_scene_index, include_scene_name, exclude_scene_regex) -> SceneSelection:
    """Validate and filter scenes using BioIO as the authoritative source."""
    xml_by_index = {s.index: s for s in scenes_xml}
    missing_metadata = [idx for idx in range(len(bioio_scenes)) if idx not in xml_by_index]
    if missing_metadata:
        raise ValueError(
            "OME-XML metadata did not describe all BioIO scenes; missing indices="
            f"{missing_metadata}"
        )

    include_scene_index = list(include_scene_index or [])
    include_scene_name = list(include_scene_name or [])
    exclude_scene_regex = list(exclude_scene_regex or [])
    bioio_names_are_placeholders = all(_is_placeholder_bioio_name(name, idx) for idx, name in enumerate(bioio_scenes))

    missing_indices = sorted(idx for idx in include_scene_index if idx < 0 or idx >= len(bioio_scenes))
    if missing_indices:
        raise ValueError(
            "Requested scene indices were not found by BioIO: "
            f"{missing_indices}; available indices=0..{len(bioio_scenes) - 1}"
        )

    available_name_counts: dict[str, int] = {}
    available_names = [xml_by_index[idx].name for idx in range(len(bioio_scenes))] if bioio_names_are_placeholders else list(bioio_scenes)
    for name in available_names:
        available_name_counts[name] = available_name_counts.get(name, 0) + 1
    missing_names = sorted({name for name in include_scene_name if available_name_counts.get(name, 0) == 0})
    if missing_names:
        raise ValueError(
            "Requested scene names were not found by BioIO: "
            f"{missing_names}; available names={available_names}"
        )

    selected_indices: set[int] = set()
    out = []
    for idx, bioio_name in enumerate(bioio_scenes):
        s = xml_by_index[idx]
        if not bioio_names_are_placeholders and s.name != bioio_name:
            raise ValueError(
                f"Scene mismatch at index {idx}: OME-XML name={s.name!r}, BioIO name={bioio_name!r}"
            )
        effective_name = s.name if bioio_names_are_placeholders else bioio_name
        if not _is_included(
            idx=s.index,
            name=effective_name,
            include_scene_index=include_scene_index,
            include_scene_name=include_scene_name,
            exclude_scene_regex=exclude_scene_regex,
        ):
            logger.debug("Excluded scene %s because it matched an exclusion pattern", s.name)
            continue
        out.append(s)
        selected_indices.add(s.index)

    wanted_indices = {
        idx
        for idx, bioio_name in enumerate(bioio_scenes)
        if _is_included(
            idx=idx,
            name=xml_by_index[idx].name if bioio_names_are_placeholders else bioio_name,
            include_scene_index=include_scene_index,
            include_scene_name=include_scene_name,
            exclude_scene_regex=exclude_scene_regex,
        )
    }
    if selected_indices != wanted_indices:
        raise ValueError(
            "Scene selection was incomplete; wanted indices="
            f"{sorted(wanted_indices)} treated={sorted(selected_indices)}"
        )

    return SceneSelection(
        selected=out,
        total_count=len(bioio_scenes),
        excluded_count=len(bioio_scenes) - len(out),
    )


def _is_placeholder_bioio_name(name: str, idx: int) -> bool:
    """Return whether BioIO exposed a generic index-based fallback scene name."""
    return bool(re.fullmatch(rf"Image:{idx}", name))


def _is_included(*, idx: int, name: str, include_scene_index, include_scene_name, exclude_scene_regex) -> bool:
    """Return whether one scene passed the caller's include/exclude filters."""
    if include_scene_index and idx not in include_scene_index:
        return False
    if include_scene_name and name not in include_scene_name:
        return False
    if exclude_scene_regex and any(re.search(rx, name) for rx in exclude_scene_regex):
        return False
    return True
