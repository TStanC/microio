"""Datamodels shared by writer and reader APIs."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class AxisResolution:
    """Resolved spacing or timing value for one axis with provenance.

    Attributes
    ----------
    axis:
        Logical axis name such as ``x``, ``y``, ``z``, or ``t``.
    value:
        Numeric sampling interval for that axis.
    unit_normalized:
        Normalized unit token suitable for downstream OME-Zarr metadata.
    unit_raw:
        Raw unit string encountered in the source metadata.
    source:
        Metadata source that produced the value, for example ``pixels``,
        ``plane_delta``, ``original_metadata``, or ``fallback``.
    confidence:
        Qualitative confidence label for downstream reporting.
    fallback:
        Whether the value is a synthetic fallback rather than a direct metadata
        measurement or inference.
    warning_code:
        Optional machine-readable warning or normalization code.
    """

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
    """Per-scene conversion report item.

    Attributes
    ----------
    scene_index:
        Zero-based scene index in the source file.
    scene_id:
        Filesystem-safe scene identifier written into the OME-Zarr output.
    converted:
        Whether image and metadata writing completed successfully.
    warnings:
        Per-scene warnings or exception messages gathered during conversion.
    axis_resolution:
        Resolved axis sampling metadata for the scene.
    """

    scene_index: int
    scene_id: str
    converted: bool
    warnings: list[str] = field(default_factory=list)
    axis_resolution: dict[str, AxisResolution] = field(default_factory=dict)


@dataclass
class ConversionReport:
    """Top-level conversion report returned by writer functions.

    Attributes
    ----------
    input_path:
        Source microscopy file path.
    output_path:
        Root path of the generated OME-Zarr dataset.
    target_ngff:
        NGFF version string written into the output.
    scene_reports:
        Per-scene conversion results in processing order.
    errors:
        Fatal per-scene errors that prevented conversion.
    warnings:
        Dataset-level non-fatal warnings.
    fallback_events:
        Structured list of axis-resolution fallback events.
    """

    input_path: Path
    output_path: Path
    target_ngff: str
    scene_reports: list[SceneReport] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    fallback_events: list[dict[str, str]] = field(default_factory=list)


@dataclass
class DatasetHandle:
    """Opened Zarr dataset handle with convenience accessors.

    Attributes
    ----------
    path:
        Filesystem path to the dataset root.
    root:
        Readable Zarr group object for the dataset root.
    """

    path: Path
    root: Any

    def list_scenes(self) -> list[str]:
        """List scene groups in the dataset root, excluding ``OME``.

        Returns
        -------
        list[str]
            Sorted scene identifiers.
        """
        from microio.reader.metadata import list_scenes

        return list_scenes(self)

    def read_scene_metadata(self, scene_id: str) -> dict:
        """Return attributes for one scene group.

        Parameters
        ----------
        scene_id:
            Scene identifier under the dataset root.

        Returns
        -------
        dict
            Scene metadata dictionary.
        """
        from microio.reader.metadata import scene_metadata

        return scene_metadata(self, scene_id)

    def read_root_metadata(self) -> dict:
        """Return root-group attributes.

        Returns
        -------
        dict
            Root metadata dictionary.
        """
        from microio.reader.metadata import root_metadata

        return root_metadata(self)

    def read_table(self, scene_id: str, table_name: str) -> dict:
        """Read one named table from ``scene/tables``.

        Parameters
        ----------
        scene_id:
            Scene identifier containing the table.
        table_name:
            Table name under ``scene/tables``.

        Returns
        -------
        dict
            Mapping from column name to NumPy array.
        """
        from microio.reader.tables import read_table

        return read_table(self, scene_id, table_name)

    def read_table_metadata(self, scene_id: str, table_name: str) -> dict:
        """Read attributes for one named table from ``scene/tables``.

        Parameters
        ----------
        scene_id:
            Scene identifier containing the table.
        table_name:
            Table name under ``scene/tables``.

        Returns
        -------
        dict
            Table metadata dictionary.
        """
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
        """Construct a synthetic per-plane position vector for one axis.

        Parameters
        ----------
        scene_id:
            Scene identifier containing the table.
        axis:
            Axis name for which positions should be built.
        table_name:
            Source table name under ``scene/tables``.
        spacing:
            Optional sampling interval override.
        origin:
            Offset added to the generated coordinate vector.
        order:
            Optional explicit logical-order vector.

        Returns
        -------
        Any
            One-dimensional coordinate vector aligned to table rows.
        """
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
        """Read the ``microio`` extension block from scene metadata.

        Parameters
        ----------
        scene_id:
            Scene identifier under the dataset root.

        Returns
        -------
        dict
            Stored ``microio`` metadata or an empty dictionary.
        """
        from microio.reader.extras import read_microio_extras

        return read_microio_extras(self, scene_id)

    def read_scene_array(self, scene_id: str, level: str = "0"):
        """Return the Zarr array handle for one scene pyramid level.

        Parameters
        ----------
        scene_id:
            Scene identifier under the dataset root.
        level:
            Pyramid level name, defaulting to the base-resolution level ``"0"``.

        Returns
        -------
        Any
            Zarr array handle for deferred or eager indexing.
        """
        return self.root[scene_id][level]
