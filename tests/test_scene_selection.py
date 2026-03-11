from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from pathlib import Path
import sys

import pytest


MODULE_PATH = Path(__file__).resolve().parents[1] / "microio" / "writer" / "scene_selection.py"
SPEC = importlib.util.spec_from_file_location("scene_selection_under_test", MODULE_PATH)
scene_selection = importlib.util.module_from_spec(SPEC)
assert SPEC is not None and SPEC.loader is not None
sys.modules[SPEC.name] = scene_selection
SPEC.loader.exec_module(scene_selection)


@dataclass(frozen=True)
class Scene:
    index: int
    name: str


def test_select_scenes_reports_counts_and_filtered_list():
    scenes_xml = [Scene(0, "A"), Scene(1, "Preview"), Scene(2, "B")]

    result = scene_selection.select_scenes(
        scenes_xml,
        bioio_scenes=["A", "Preview", "B"],
        include_scene_index=None,
        include_scene_name=None,
        exclude_scene_regex=["Preview"],
    )

    assert [scene.name for scene in result.selected] == ["A", "B"]
    assert result.total_count == 3
    assert result.excluded_count == 1


def test_select_scenes_errors_when_requested_name_is_missing_from_bioio():
    scenes_xml = [Scene(0, "A"), Scene(1, "B")]

    with pytest.raises(ValueError, match="Requested scene names were not found by BioIO"):
        scene_selection.select_scenes(
            scenes_xml,
            bioio_scenes=["A", "B"],
            include_scene_index=None,
            include_scene_name=["C"],
            exclude_scene_regex=None,
        )


def test_select_scenes_errors_when_bioio_and_xml_scene_names_disagree():
    scenes_xml = [Scene(0, "A"), Scene(1, "B")]

    with pytest.raises(ValueError, match="Scene mismatch at index 1"):
        scene_selection.select_scenes(
            scenes_xml,
            bioio_scenes=["A", "C"],
            include_scene_index=None,
            include_scene_name=None,
            exclude_scene_regex=None,
        )


def test_select_scenes_errors_when_bioio_has_scene_missing_from_xml():
    scenes_xml = [Scene(0, "A")]

    with pytest.raises(ValueError, match="missing indices=\\[1\\]"):
        scene_selection.select_scenes(
            scenes_xml,
            bioio_scenes=["A", "B"],
            include_scene_index=None,
            include_scene_name=None,
            exclude_scene_regex=None,
        )


def test_select_scenes_accepts_placeholder_bioio_names_and_filters_by_xml_name():
    scenes_xml = [Scene(0, "ClimateDataGraph"), Scene(1, "BigCircle1"), Scene(2, "BigCircle2")]

    result = scene_selection.select_scenes(
        scenes_xml,
        bioio_scenes=["Image:0", "Image:1", "Image:2"],
        include_scene_index=[1, 2],
        include_scene_name=None,
        exclude_scene_regex=["Climate"],
    )

    assert [scene.index for scene in result.selected] == [1, 2]
    assert result.total_count == 3
    assert result.excluded_count == 1


def test_select_scenes_accepts_placeholder_bioio_names_for_include_scene_name():
    scenes_xml = [Scene(0, "ClimateDataGraph"), Scene(1, "BigCircle1"), Scene(2, "BigCircle2")]

    result = scene_selection.select_scenes(
        scenes_xml,
        bioio_scenes=["Image:0", "Image:1", "Image:2"],
        include_scene_index=None,
        include_scene_name=["BigCircle2"],
        exclude_scene_regex=None,
    )

    assert [scene.index for scene in result.selected] == [2]
    assert result.total_count == 3
    assert result.excluded_count == 2


def test_select_scenes_accepts_placeholder_bioio_names_for_exclude_scene_regex():
    scenes_xml = [Scene(0, "ClimateDataGraph"), Scene(1, "BigCircle1"), Scene(2, "BigCircle2")]

    result = scene_selection.select_scenes(
        scenes_xml,
        bioio_scenes=["Image:0", "Image:1", "Image:2"],
        include_scene_index=None,
        include_scene_name=None,
        exclude_scene_regex=["Climate"],
    )

    assert [scene.index for scene in result.selected] == [1, 2]
    assert result.total_count == 3
    assert result.excluded_count == 1
