"""Datamodels shared by writer and reader APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AxisResolution:
    """Resolved spacing/time value for one axis with provenance."""

    axis: str
    value: float
    unit_normalized: str | None
    unit_raw: str | None
    source: str
    confidence: str
    fallback: bool
    warning_code: str | None = None


@dataclass
class SceneReport:
    """Per-scene conversion report item."""

    scene_index: int
    scene_id: str
    converted: bool
    warnings: list[str] = field(default_factory=list)
    axis_resolution: dict[str, AxisResolution] = field(default_factory=dict)


@dataclass
class ConversionReport:
    """Top-level conversion report returned by writer functions."""

    input_path: Path
    output_path: Path
    target_ngff: str
    scene_reports: list[SceneReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_events: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DatasetHandle:
    """Opened zarr dataset handle with convenience accessors."""

    path: Path
    root: Any

    def list_scenes(self) -> list[str]:
        """List scene groups in dataset root, excluding ``OME``."""
        from microio.reader.metadata import list_scenes

        return list_scenes(self)

    def read_scene_metadata(self, scene_id: str) -> dict:
        """Return attrs for one scene group."""
        from microio.reader.metadata import scene_metadata

        return scene_metadata(self, scene_id)

    def read_root_metadata(self) -> dict:
        """Return root attrs."""
        from microio.reader.metadata import root_metadata

        return root_metadata(self)

    def read_table(self, scene_id: str, table_name: str) -> dict:
        """Read one named table from ``scene/tables``."""
        from microio.reader.tables import read_table

        return read_table(self, scene_id, table_name)

    def read_table_metadata(self, scene_id: str, table_name: str) -> dict:
        """Read attrs for one named table from ``scene/tables``."""
        from microio.reader.tables import read_table_metadata

        return read_table_metadata(self, scene_id, table_name)

    def build_axis_positions(
        self,
        scene_id: str,
        axis: str,
        *,
        table_name: str = "axes_trajectory",
        spacing: float | None = None,
        origin: float = 0.0,
        order: Any = None,
    ):
        """Construct an artificial per-plane position vector for one axis."""
        from microio.reader.tables import build_axis_positions

        return build_axis_positions(
            self,
            scene_id,
            axis,
            table_name=table_name,
            spacing=spacing,
            origin=origin,
            order=order,
        )

    def read_microio_extras(self, scene_id: str) -> dict:
        """Read microio extension block from scene attrs."""
        from microio.reader.extras import read_microio_extras

        return read_microio_extras(self, scene_id)

    def read_scene_array(self, scene_id: str, level: str = "0"):
        """Return zarr array handle for one scene pyramid level."""
        return self.root[scene_id][level]
