"""Shared datamodels for the reader and enrichment APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationMessage:
    """Structured validation or repair diagnostic."""

    level: str
    code: str
    message: str


@dataclass(frozen=True)
class AxisState:
    """Resolved scene-axis state after multiscale inspection."""

    axis: str
    value: float | None
    unit: str | None
    raw_unit: str | None
    source: str
    placeholder: bool
    repaired: bool
    confidence: str
    warning_code: str | None = None


@dataclass(frozen=True)
class MultiscaleLevel:
    """One multiscale level with the parsed scale vector."""

    path: str
    scale: list[float]


@dataclass(frozen=True)
class SceneOmeMetadata:
    """Normalized scene metadata extracted from OME-XML."""

    index: int
    name: str
    size_t: int
    size_c: int
    size_z: int
    size_y: int
    size_x: int
    physical_size_x: float | None
    physical_size_x_unit: str | None
    physical_size_y: float | None
    physical_size_y_unit: str | None
    physical_size_z: float | None
    physical_size_z_unit: str | None
    time_increment: float | None
    time_increment_unit: str | None
    planes: list[dict[str, str | None]]


@dataclass
class PlaneTableReport:
    """Result of building or loading a plane table."""

    scene_id: str
    table_name: str
    row_count: int
    persisted: bool
    warnings: list[ValidationMessage] = field(default_factory=list)


@dataclass
class RepairReport:
    """Result of validating and optionally repairing one scene."""

    scene_id: str
    persisted: bool
    axis_states: dict[str, AxisState]
    warnings: list[ValidationMessage] = field(default_factory=list)
    errors: list[ValidationMessage] = field(default_factory=list)


@dataclass
class DatasetHandle:
    """Opened OME-Zarr dataset handle with enrichment helpers."""

    path: Path
    root: Any
    mode: str = "r"

    def list_scenes(self) -> list[str]:
        from microio.reader.metadata import list_scenes

        return list_scenes(self)

    def read_root_metadata(self) -> dict:
        from microio.reader.metadata import root_metadata

        return root_metadata(self)

    def read_scene_metadata(self, scene_id: str, *, corrected: bool = True) -> dict:
        from microio.reader.metadata import scene_metadata

        return scene_metadata(self, scene_id, corrected=corrected)

    def read_multiscale_metadata(self, scene_id: str) -> dict:
        from microio.reader.metadata import multiscale_metadata

        return multiscale_metadata(self, scene_id)

    def read_ome_xml(self) -> str:
        from microio.reader.metadata import read_ome_xml

        return read_ome_xml(self)

    def read_scene_ome_metadata(self, scene_id: str) -> SceneOmeMetadata:
        from microio.reader.metadata import scene_ome_metadata

        return scene_ome_metadata(self, scene_id)

    def read_original_metadata(self) -> dict[str, str]:
        from microio.reader.metadata import original_metadata

        return original_metadata(self)

    def inspect_axis_metadata(self, scene_id: str) -> RepairReport:
        from microio.reader.repair import inspect_axis_metadata

        return inspect_axis_metadata(self, scene_id)

    def repair_axis_metadata(self, scene_id: str, *, persist: bool = True) -> RepairReport:
        from microio.reader.repair import repair_axis_metadata

        return repair_axis_metadata(self, scene_id, persist=persist)

    def load_plane_table(self, scene_id: str, table_name: str = "axes_trajectory") -> dict[str, Any]:
        from microio.reader.tables import load_plane_table

        return load_plane_table(self, scene_id, table_name=table_name)

    def build_plane_table(
        self,
        scene_id: str,
        *,
        table_name: str = "axes_trajectory",
        persist: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        from microio.reader.tables import build_plane_table

        return build_plane_table(self, scene_id, table_name=table_name, persist=persist)

    def ensure_plane_table(
        self,
        scene_id: str,
        *,
        table_name: str = "axes_trajectory",
        rebuild: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        from microio.reader.tables import ensure_plane_table

        return ensure_plane_table(self, scene_id, table_name=table_name, rebuild=rebuild)

    def read_table_metadata(self, scene_id: str, table_name: str = "axes_trajectory") -> dict:
        from microio.reader.tables import read_table_metadata

        return read_table_metadata(self, scene_id, table_name)

    def read_microio_extras(self, scene_id: str) -> dict:
        from microio.reader.extras import read_microio_extras

        return read_microio_extras(self, scene_id)

    def read_scene_array(self, scene_id: str, level: str = "0"):
        return self.root[scene_id][level]
