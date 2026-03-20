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
class SceneRef:
    """Canonical scene identity inside one opened dataset."""

    id: str
    index: int
    name: str
    group_path: str
    ome_index: int | None = None
    duplicate_name_count: int = 1


@dataclass(frozen=True)
class LevelRef:
    """Validated multiscale level description for one scene."""

    scene_id: str
    level_index: int
    path: str
    shape: tuple[int, ...]
    dtype: str
    scale: tuple[float, ...]
    axis_names: tuple[str, ...]
    axis_units: tuple[str | None, ...]


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
    planes: tuple[dict[str, str | None], ...]


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
class TableWriteReport:
    """Result of writing or appending one scene table."""

    scene_id: str
    table_name: str
    row_count: int
    column_names: list[str]
    persisted: bool
    appended: bool = False


@dataclass
class LabelWriteReport:
    """Result of writing one label image."""

    scene_id: str
    label_name: str
    level_path: str
    shape: tuple[int, ...]
    dtype: str
    persisted: bool


@dataclass
class RoiWriteReport:
    """Result of writing one ROI image group."""

    scene_id: str
    roi_name: str
    level_path: str
    shape: tuple[int, ...]
    persisted: bool


@dataclass
class DataFlowReport:
    """Consistency report for scene identity, metadata, and array access."""

    scene: SceneRef
    levels: list[LevelRef]
    warnings: list[ValidationMessage] = field(default_factory=list)
    errors: list[ValidationMessage] = field(default_factory=list)


@dataclass(frozen=True)
class SceneAccessor:
    """Convenience wrapper for one resolved scene."""

    dataset: "DatasetHandle"
    ref: SceneRef

    def metadata(self, *, corrected: bool = True) -> dict:
        return self.dataset.read_scene_metadata(self.ref.id, corrected=corrected)

    def multiscale_metadata(self) -> dict:
        return self.dataset.read_multiscale_metadata(self.ref.id)

    def ome_metadata(self) -> SceneOmeMetadata:
        return self.dataset.read_scene_ome_metadata(self.ref.id)

    def levels(self) -> list[LevelRef]:
        return self.dataset.list_levels(self.ref.id)

    def level(self, level: int | str = 0) -> LevelRef:
        return self.dataset.level_ref(self.ref.id, level)

    def array(self, level: int | str = 0):
        return self.dataset.read_level(self.ref.id, level)

    def zarr_array(self, level: int | str = 0):
        return self.dataset.read_level_zarr(self.ref.id, level)

    def numpy_array(self, level: int | str = 0):
        return self.dataset.read_level_numpy(self.ref.id, level)


@dataclass
class DatasetHandle:
    """Opened OME-Zarr dataset handle with enrichment helpers."""

    path: Path
    root: Any
    mode: str = "r"
    _scene_refs_cache: list[SceneRef] | None = field(default=None, init=False, repr=False)
    _level_refs_cache: dict[str, list[LevelRef]] = field(default_factory=dict, init=False, repr=False)
    _raw_scene_metadata_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _ome_document_cache: Any | None = field(default=None, init=False, repr=False)

    def invalidate_caches(self, scene_id: str | None = None) -> None:
        self._scene_refs_cache = None
        if scene_id is None:
            self._level_refs_cache.clear()
            self._raw_scene_metadata_cache.clear()
        else:
            self._level_refs_cache.pop(str(scene_id), None)
            self._raw_scene_metadata_cache.pop(str(scene_id), None)

    def list_scene_refs(self) -> list[SceneRef]:
        from microio.reader.metadata import list_scene_refs

        return list_scene_refs(self)

    def list_scenes(self) -> list[str]:
        from microio.reader.metadata import list_scenes

        return list_scenes(self)

    def scene_ref(self, scene: int | str) -> SceneRef:
        from microio.reader.metadata import scene_ref

        return scene_ref(self, scene)

    def get_scene(self, scene: int | str) -> SceneAccessor:
        ref = self.scene_ref(scene)
        return SceneAccessor(dataset=self, ref=ref)

    def classify_scene_reference(self, value: int | str) -> str:
        from microio.reader.metadata import classify_scene_reference

        return classify_scene_reference(self, value)

    def is_scene_id(self, value: str) -> bool:
        from microio.reader.metadata import is_scene_id

        return is_scene_id(self, value)

    def is_scene_index(self, value: int) -> bool:
        from microio.reader.metadata import is_scene_index

        return is_scene_index(self, value)

    def scene_id_to_index(self, scene_id: str) -> int:
        from microio.reader.metadata import scene_id_to_index

        return scene_id_to_index(self, scene_id)

    def scene_index_to_id(self, index: int) -> str:
        from microio.reader.metadata import scene_index_to_id

        return scene_index_to_id(self, index)

    def scene_name_matches(self, name: str) -> list[SceneRef]:
        from microio.reader.metadata import scene_name_matches

        return scene_name_matches(self, name)

    def read_root_metadata(self) -> dict:
        from microio.reader.metadata import root_metadata

        return root_metadata(self)

    def read_scene_metadata(self, scene: int | str, *, corrected: bool = True) -> dict:
        from microio.reader.metadata import scene_metadata

        return scene_metadata(self, scene, corrected=corrected)

    def read_multiscale_metadata(self, scene: int | str) -> dict:
        from microio.reader.metadata import multiscale_metadata

        return multiscale_metadata(self, scene)

    def read_ome_xml(self) -> str:
        from microio.reader.metadata import read_ome_xml

        return read_ome_xml(self)

    def read_scene_ome_metadata(self, scene: int | str) -> SceneOmeMetadata:
        from microio.reader.metadata import scene_ome_metadata

        return scene_ome_metadata(self, scene)

    def read_original_metadata(self) -> dict[str, str]:
        from microio.reader.metadata import original_metadata

        return original_metadata(self)

    def list_levels(self, scene: int | str) -> list[LevelRef]:
        from microio.reader.metadata import list_levels

        return list_levels(self, scene)

    def level_ref(self, scene: int | str, level: int | str) -> LevelRef:
        from microio.reader.metadata import level_ref

        return level_ref(self, scene, level)

    def read_level(self, scene: int | str, level: int | str = 0):
        from microio.reader.metadata import read_level

        return read_level(self, scene, level)

    def read_level_zarr(self, scene: int | str, level: int | str = 0):
        from microio.reader.metadata import read_level_zarr

        return read_level_zarr(self, scene, level)

    def read_level_numpy(self, scene: int | str, level: int | str = 0):
        from microio.reader.metadata import read_level_numpy

        return read_level_numpy(self, scene, level)

    def validate_scene_data_flow(self, scene: int | str) -> DataFlowReport:
        from microio.reader.metadata import validate_scene_data_flow

        return validate_scene_data_flow(self, scene)

    def inspect_axis_metadata(self, scene: int | str) -> RepairReport:
        from microio.reader.repair import inspect_axis_metadata

        return inspect_axis_metadata(self, scene)

    def repair_axis_metadata(self, scene: int | str, *, persist: bool = True) -> RepairReport:
        from microio.reader.repair import repair_axis_metadata

        return repair_axis_metadata(self, scene, persist=persist)

    def load_plane_table(self, scene: int | str, table_name: str = "axes_trajectory") -> dict[str, Any]:
        from microio.reader.tables import load_plane_table

        return load_plane_table(self, scene, table_name=table_name)

    def build_plane_table(
        self,
        scene: int | str,
        *,
        table_name: str = "axes_trajectory",
        persist: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        from microio.reader.tables import build_plane_table

        return build_plane_table(self, scene, table_name=table_name, persist=persist)

    def ensure_plane_table(
        self,
        scene: int | str,
        *,
        table_name: str = "axes_trajectory",
        rebuild: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        from microio.reader.tables import ensure_plane_table

        return ensure_plane_table(self, scene, table_name=table_name, rebuild=rebuild)

    def read_table_metadata(self, scene: int | str, table_name: str = "axes_trajectory") -> dict:
        from microio.reader.tables import read_table_metadata

        return read_table_metadata(self, scene, table_name)

    def read_microio_extras(self, scene: int | str) -> dict:
        from microio.reader.extras import read_microio_extras

        return read_microio_extras(self, scene)

    def write_table(
        self,
        scene: int | str,
        name: str,
        data: Any,
        *,
        attrs: dict[str, Any] | None = None,
        overwrite: bool = False,
        append: bool = False,
        chunk_length: int | None = None,
    ) -> TableWriteReport:
        from microio.writer.tables import write_table

        return write_table(
            self,
            scene,
            name,
            data,
            attrs=attrs,
            overwrite=overwrite,
            append=append,
            chunk_length=chunk_length,
        )

    def write_label_image(
        self,
        scene: int | str,
        name: str,
        data: Any,
        *,
        source_level: int | str = 0,
        chunks: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        attrs: dict[str, Any] | None = None,
        colors: list[dict[str, Any]] | None = None,
        properties: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
        threads: int | None = None,
    ) -> LabelWriteReport:
        from microio.writer.images import write_label_image

        return write_label_image(
            self,
            scene,
            name,
            data,
            source_level=source_level,
            chunks=chunks,
            dtype=dtype,
            attrs=attrs,
            colors=colors,
            properties=properties,
            overwrite=overwrite,
            threads=threads,
        )

    def write_roi(
        self,
        scene: int | str,
        name: str,
        slices: dict[str, Any],
        *,
        source_level: int | str = 0,
        chunks: tuple[int, ...] | None = None,
        attrs: dict[str, Any] | None = None,
        overwrite: bool = False,
        threads: int | None = None,
    ) -> RoiWriteReport:
        from microio.writer.images import write_roi

        return write_roi(
            self,
            scene,
            name,
            slices,
            source_level=source_level,
            chunks=chunks,
            attrs=attrs,
            overwrite=overwrite,
            threads=threads,
        )
