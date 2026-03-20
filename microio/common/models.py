"""Shared datamodels for the reader and enrichment APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ValidationMessage:
    """Structured validation or repair diagnostic.

    Attributes
    ----------
    level:
        Severity level such as ``"warning"`` or ``"error"``.
    code:
        Stable machine-readable code for programmatic checks in tests or CLI
        consumers.
    message:
        Human-readable explanation of the condition.
    """

    level: str
    code: str
    message: str


@dataclass(frozen=True)
class AxisState:
    """Resolved scene-axis state after multiscale inspection.

    Attributes
    ----------
    axis:
        Axis name from the supported ``("t", "c", "z", "y", "x")`` order.
    value:
        Scalar spacing value currently associated with the axis at level ``0``.
    unit:
        Normalized unit token such as ``"micrometer"`` or ``"second"``.
    raw_unit:
        Original unit token observed in the dataset before normalization.
    source:
        Provenance string describing where the value came from, for example
        ``"zarr"``, ``"Pixels.PhysicalSizeZ"``, or ``"Plane.PositionZ"``.
    placeholder:
        Whether the current axis metadata looks like a placeholder that may
        need repair.
    repaired:
        Whether the state reflects a microio repair decision rather than the
        original stored metadata.
    confidence:
        Qualitative confidence label such as ``"high"`` or ``"medium"``.
    warning_code:
        Optional normalization warning code associated with ``unit``.
    """

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
    """One multiscale level with the parsed scale vector.

    Attributes
    ----------
    path:
        Relative array path from the scene group, for example ``"0"`` or
        ``"1"``.
    scale:
        Parsed scale vector in dataset axis order.
    """

    path: str
    scale: list[float]


@dataclass(frozen=True)
class SceneRef:
    """Canonical scene identity inside one opened dataset.

    Attributes
    ----------
    id:
        Canonical Zarr child key for the scene.
    index:
        Zero-based scene position in stable dataset order.
    name:
        Human-readable scene name derived from multiscale metadata.
    group_path:
        Group path for the scene within the root Zarr store.
    ome_index:
        Matched OME image index when the sidecar OME-XML can be resolved
        safely.
    duplicate_name_count:
        Number of scenes in the dataset that share ``name``.
    """

    id: str
    index: int
    name: str
    group_path: str
    ome_index: int | None = None
    duplicate_name_count: int = 1


@dataclass(frozen=True)
class LevelRef:
    """Validated multiscale level description for one scene.

    Attributes
    ----------
    scene_id:
        Canonical scene id that owns the level.
    level_index:
        Zero-based multiscale level index.
    path:
        Dataset path string from the multiscales metadata.
    shape:
        Array shape validated against the Zarr store.
    dtype:
        String form of the underlying array dtype.
    scale:
        Parsed coordinate scale vector for the level.
    axis_names:
        Axis order validated against the library contract.
    axis_units:
        Optional unit string for each axis.
    """

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
    """Normalized scene metadata extracted from OME-XML.

    This dataclass contains the subset of OME image metadata used for
    validation, plane-table generation, and conservative axis repair.

    Attributes
    ----------
    index:
        Image index within the parsed OME document.
    name:
        OME ``Image`` name.
    size_t, size_c, size_z, size_y, size_x:
        Pixel sizes from the OME ``Pixels`` element.
    physical_size_x, physical_size_y, physical_size_z:
        Optional scalar physical pixel sizes from OME.
    physical_size_x_unit, physical_size_y_unit, physical_size_z_unit:
        Raw OME unit tokens for the physical pixel sizes.
    time_increment:
        Optional scalar OME time increment.
    time_increment_unit:
        Raw OME unit token for ``time_increment``.
    planes:
        Per-plane metadata rows as parsed dictionaries containing fields such
        as ``TheT``, ``TheC``, ``TheZ``, ``DeltaT``, and ``PositionZ``.
    """

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
    """Result of building or loading a plane table.

    Attributes
    ----------
    scene_id:
        Canonical scene id that owns the table.
    table_name:
        Scene-local table name, usually ``"axes_trajectory"``.
    row_count:
        Number of rows in the loaded or generated table.
    persisted:
        Whether this call wrote a table into the dataset.
    warnings:
        Non-fatal diagnostics gathered while building or validating the table.
    """

    scene_id: str
    table_name: str
    row_count: int
    persisted: bool
    warnings: list[ValidationMessage] = field(default_factory=list)


@dataclass
class RepairReport:
    """Result of validating and optionally repairing one scene.

    Attributes
    ----------
    scene_id:
        Canonical scene id that was inspected or repaired.
    persisted:
        Whether the call wrote any accepted repair back into the dataset.
    axis_states:
        Final per-axis state map keyed by axis name.
    warnings:
        Non-fatal diagnostics, including unresolved placeholder metadata.
    errors:
        Fatal validation issues that prevented repair.
    """

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
    """Result of writing one label image.

    Attributes
    ----------
    scene_id:
        Canonical scene id that owns the label image.
    label_name:
        Scene-local label image name under ``labels/``.
    level_path:
        Finest written pyramid path, usually ``"0"``.
    shape:
        Shape of the payload passed to the write call.
    dtype:
        String form of the written label dtype.
    persisted:
        Whether the write completed successfully.
    written_timepoint:
        Timepoint written by :meth:`DatasetHandle.write_label_timepoint`, or
        ``None`` for whole-image writes.
    initialized:
        Whether the label image group was created in this call.
    """

    scene_id: str
    label_name: str
    level_path: str
    shape: tuple[int, ...]
    dtype: str
    persisted: bool
    written_timepoint: int | None = None
    initialized: bool = False


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
    """Convenience wrapper for one resolved scene.

    The accessor keeps a resolved :class:`SceneRef` together with its parent
    dataset handle and forwards common read operations to dataset-level APIs.
    It is primarily ergonomic sugar over ``DatasetHandle`` when the scene
    selector is already known.
    """

    dataset: "DatasetHandle"
    ref: SceneRef

    def metadata(self, *, corrected: bool = True) -> dict:
        """Return scene attributes for this accessor's scene.

        Parameters
        ----------
        corrected:
            If ``True``, overlay persisted microio repairs onto the returned
            semantic metadata view.
        """
        return self.dataset.read_scene_metadata(self.ref.id, corrected=corrected)

    def multiscale_metadata(self) -> dict:
        """Return the validated primary multiscales block for this scene."""
        return self.dataset.read_multiscale_metadata(self.ref.id)

    def ome_metadata(self) -> SceneOmeMetadata:
        """Return normalized OME metadata for this scene."""
        return self.dataset.read_scene_ome_metadata(self.ref.id)

    def levels(self) -> list[LevelRef]:
        """Return validated multiscale levels for this scene."""
        return self.dataset.list_levels(self.ref.id)

    def level(self, level: int | str = 0) -> LevelRef:
        """Resolve one level by index or path for this scene.

        Examples
        --------
        ``scene.level(0)`` resolves the finest level, while
        ``scene.level("1")`` resolves the level stored under path ``"1"``.
        """
        return self.dataset.level_ref(self.ref.id, level)

    def array(self, level: int | str = 0):
        """Return one level as a lazy Dask array."""
        return self.dataset.read_level(self.ref.id, level)

    def zarr_array(self, level: int | str = 0):
        """Return one level as the underlying Zarr array."""
        return self.dataset.read_level_zarr(self.ref.id, level)

    def numpy_array(self, level: int | str = 0):
        """Return one level eagerly materialized as a NumPy array.

        Unlike :meth:`array`, this reads the full level into memory.
        """
        return self.dataset.read_level_numpy(self.ref.id, level)


@dataclass
class DatasetHandle:
    """Opened OME-Zarr dataset handle with inspection and enrichment helpers.

    Parameters
    ----------
    path:
        Filesystem path to the opened dataset.
    root:
        Opened Zarr root group.
    mode:
        Zarr access mode used to open the dataset, usually ``"r"`` for
        inspection or ``"a"`` for repair and writer operations.

    Notes
    -----
    Most methods on this dataclass are thin forwarding wrappers around the
    richer reader and writer modules. These wrappers exist so users can stay on
    the dataset object once it has been opened.
    """

    path: Path
    root: Any
    mode: str = "r"
    _scene_refs_cache: list[SceneRef] | None = field(default=None, init=False, repr=False)
    _level_refs_cache: dict[str, list[LevelRef]] = field(default_factory=dict, init=False, repr=False)
    _raw_scene_metadata_cache: dict[str, dict[str, Any]] = field(default_factory=dict, init=False, repr=False)
    _ome_document_cache: Any | None = field(default=None, init=False, repr=False)

    def invalidate_caches(self, scene_id: str | None = None) -> None:
        """Invalidate cached scene, level, and metadata lookups.

        Parameters
        ----------
        scene_id:
            Optional scene id whose scene-local caches should be cleared. When
            omitted, all cached scene, level, and OME-backed metadata lookups
            are invalidated.
        """
        self._scene_refs_cache = None
        if scene_id is None:
            self._level_refs_cache.clear()
            self._raw_scene_metadata_cache.clear()
        else:
            self._level_refs_cache.pop(str(scene_id), None)
            self._raw_scene_metadata_cache.pop(str(scene_id), None)

    def list_scene_refs(self) -> list[SceneRef]:
        """Return canonical scene references in stable dataset order."""
        from microio.reader.metadata import list_scene_refs

        return list_scene_refs(self)

    def list_scenes(self) -> list[str]:
        """Return canonical scene ids in stable dataset order."""
        from microio.reader.metadata import list_scenes

        return list_scenes(self)

    def scene_ref(self, scene: int | str) -> SceneRef:
        """Resolve a scene by id, dataset index, or unique display name.

        Examples
        --------
        ``ds.scene_ref(0)`` resolves the first scene in dataset order,
        ``ds.scene_ref("0")`` resolves the canonical scene id, and
        ``ds.scene_ref("C555")`` resolves a unique multiscale scene name.
        """
        from microio.reader.metadata import scene_ref

        return scene_ref(self, scene)

    def get_scene(self, scene: int | str) -> SceneAccessor:
        """Return a convenience accessor bound to one resolved scene."""
        ref = self.scene_ref(scene)
        return SceneAccessor(dataset=self, ref=ref)

    def classify_scene_reference(self, value: int | str) -> str:
        """Classify a candidate scene selector without raising on misses."""
        from microio.reader.metadata import classify_scene_reference

        return classify_scene_reference(self, value)

    def is_scene_id(self, value: str) -> bool:
        """Return whether ``value`` is a canonical scene id."""
        from microio.reader.metadata import is_scene_id

        return is_scene_id(self, value)

    def is_scene_index(self, value: int) -> bool:
        """Return whether ``value`` is a valid dataset-order scene index."""
        from microio.reader.metadata import is_scene_index

        return is_scene_index(self, value)

    def scene_id_to_index(self, scene_id: str) -> int:
        """Convert a canonical scene id into its dataset-order index."""
        from microio.reader.metadata import scene_id_to_index

        return scene_id_to_index(self, scene_id)

    def scene_index_to_id(self, index: int) -> str:
        """Convert a dataset-order scene index into its canonical scene id."""
        from microio.reader.metadata import scene_index_to_id

        return scene_index_to_id(self, index)

    def scene_name_matches(self, name: str) -> list[SceneRef]:
        """Return all scenes whose display name matches ``name`` exactly."""
        from microio.reader.metadata import scene_name_matches

        return scene_name_matches(self, name)

    def read_root_metadata(self) -> dict:
        """Return root-group metadata as plain Python objects."""
        from microio.reader.metadata import root_metadata

        return root_metadata(self)

    def read_scene_metadata(self, scene: int | str, *, corrected: bool = True) -> dict:
        """Return one scene's semantic metadata view.

        Parameters
        ----------
        scene:
            Scene selector accepted by :meth:`scene_ref`.
        corrected:
            If ``True``, apply persisted microio repairs before returning the
            metadata.
        """
        from microio.reader.metadata import scene_metadata

        return scene_metadata(self, scene, corrected=corrected)

    def read_multiscale_metadata(self, scene: int | str) -> dict:
        """Return the validated primary multiscales block for one scene."""
        from microio.reader.metadata import multiscale_metadata

        return multiscale_metadata(self, scene)

    def read_ome_xml(self) -> str:
        """Return the raw dataset-level sidecar OME-XML text."""
        from microio.reader.metadata import read_ome_xml

        return read_ome_xml(self)

    def read_scene_ome_metadata(self, scene: int | str) -> SceneOmeMetadata:
        """Return normalized OME metadata for one scene.

        This is the parsed OME view, not the raw semantic scene attrs returned
        by :meth:`read_scene_metadata`.
        """
        from microio.reader.metadata import scene_ome_metadata

        return scene_ome_metadata(self, scene)

    def read_original_metadata(self) -> dict[str, str]:
        """Return the OME ``OriginalMetadata`` key-value mapping."""
        from microio.reader.metadata import original_metadata

        return original_metadata(self)

    def list_levels(self, scene: int | str) -> list[LevelRef]:
        """Return validated multiscale levels for one scene."""
        from microio.reader.metadata import list_levels

        return list_levels(self, scene)

    def level_ref(self, scene: int | str, level: int | str) -> LevelRef:
        """Resolve one multiscale level by index or path."""
        from microio.reader.metadata import level_ref

        return level_ref(self, scene, level)

    def read_level(self, scene: int | str, level: int | str = 0):
        """Return one image level as a lazy Dask array.

        Use this method when downstream computation can stay lazy. For eager
        reads, use :meth:`read_level_numpy`.
        """
        from microio.reader.metadata import read_level

        return read_level(self, scene, level)

    def read_level_zarr(self, scene: int | str, level: int | str = 0):
        """Return one image level as the underlying Zarr array."""
        from microio.reader.metadata import read_level_zarr

        return read_level_zarr(self, scene, level)

    def read_level_numpy(self, scene: int | str, level: int | str = 0):
        """Return one image level eagerly materialized as a NumPy array."""
        from microio.reader.metadata import read_level_numpy

        return read_level_numpy(self, scene, level)

    def validate_scene_data_flow(self, scene: int | str) -> DataFlowReport:
        """Validate scene identity, multiscale metadata, and OME consistency."""
        from microio.reader.metadata import validate_scene_data_flow

        return validate_scene_data_flow(self, scene)

    def inspect_axis_metadata(self, scene: int | str) -> RepairReport:
        """Inspect axis metadata for one scene without mutating the dataset."""
        from microio.reader.repair import inspect_axis_metadata

        return inspect_axis_metadata(self, scene)

    def repair_axis_metadata(self, scene: int | str, *, persist: bool = True) -> RepairReport:
        """Repair trustworthy axis and channel-window metadata for one scene.

        Parameters
        ----------
        scene:
            Scene selector accepted by :meth:`scene_ref`.
        persist:
            If ``True``, accepted repairs are written back into the dataset.
            The handle must have been opened in append mode.
        """
        from microio.reader.repair import repair_axis_metadata

        return repair_axis_metadata(self, scene, persist=persist)

    def load_plane_table(self, scene: int | str, table_name: str = "axes_trajectory") -> dict[str, Any]:
        """Load a persisted plane table into eager NumPy columns.

        The default table stores one row per ``(t, c, z)`` plane with columns
        such as ``the_t``, ``the_c``, ``the_z``, and ``positioners_z``.
        """
        from microio.reader.tables import load_plane_table

        return load_plane_table(self, scene, table_name=table_name)

    def build_plane_table(
        self,
        scene: int | str,
        *,
        table_name: str = "axes_trajectory",
        persist: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        """Build a plane table from OME metadata, optionally persisting it."""
        from microio.reader.tables import build_plane_table

        return build_plane_table(self, scene, table_name=table_name, persist=persist)

    def ensure_plane_table(
        self,
        scene: int | str,
        *,
        table_name: str = "axes_trajectory",
        rebuild: bool = False,
    ) -> tuple[dict[str, Any], PlaneTableReport]:
        """Load a compatible plane table or rebuild it when required."""
        from microio.reader.tables import ensure_plane_table

        return ensure_plane_table(self, scene, table_name=table_name, rebuild=rebuild)

    def read_table_metadata(self, scene: int | str, table_name: str = "axes_trajectory") -> dict:
        """Read stored metadata attrs for one scene-local table."""
        from microio.reader.tables import read_table_metadata

        return read_table_metadata(self, scene, table_name)

    def read_microio_extras(self, scene: int | str) -> dict:
        """Read the stored ``microio`` extension block for one scene."""
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
        """Write or append a scene-local table under ``tables/<name>``.

        The ``data`` argument accepts the same forms as
        :func:`microio.writer.tables.write_table`, including mappings of
        columns, row records, flat scalar sequences, and pandas DataFrames when
        pandas is installed.
        """
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
        """Write an NGFF-style label pyramid under ``labels/<name>``.

        This method expects a level-0 label image in dataset axis order. The
        label channel axis may match the source scene or be a singleton size of
        ``1`` when one label volume applies to all source channels. Any
        coarser levels are derived from the source pyramid metadata.
        """
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

    def write_label_timepoint(
        self,
        scene: int | str,
        name: str,
        data: Any,
        *,
        timepoint: int,
        source_level: int | str = 0,
        chunks: tuple[int, ...] | None = None,
        dtype: Any | None = None,
        attrs: dict[str, Any] | None = None,
        colors: list[dict[str, Any]] | None = None,
        properties: list[dict[str, Any]] | None = None,
        overwrite: bool = False,
        overwrite_timepoint: bool = False,
        threads: int | None = None,
    ) -> LabelWriteReport:
        """Write one timepoint of an NGFF-style label image under ``labels/<name>``.

        The input must follow dataset axis order, keep all dimensions, and use
        a singleton time axis of length ``1``. This method is intended for
        caller-coordinated writes into disjoint timepoints of one label image.
        """
        from microio.writer.images import write_label_timepoint

        return write_label_timepoint(
            self,
            scene,
            name,
            data,
            timepoint=timepoint,
            source_level=source_level,
            chunks=chunks,
            dtype=dtype,
            attrs=attrs,
            colors=colors,
            properties=properties,
            overwrite=overwrite,
            overwrite_timepoint=overwrite_timepoint,
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
        """Write a single-scale ROI cutout under ``rois/<name>/0``.

        The ``slices`` mapping uses axis names such as ``"t"``, ``"z"``,
        ``"y"``, and ``"x"`` with either integer indices or ``(start, stop)``
        tuples.
        """
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
