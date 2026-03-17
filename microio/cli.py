"""Command-line interface for inspection and enrichment."""

from __future__ import annotations

import argparse
from dataclasses import fields, is_dataclass
import json
from types import MappingProxyType

from microio import open_dataset
from microio.common.logging_utils import setup_logging


def _cmd_inspect(args) -> int:
    logger = setup_logging(args.log_level)
    ds = open_dataset(args.input)
    payload = {
        "path": str(ds.path),
        "root": ds.read_root_metadata(),
        "scene_refs": [_json_ready(scene) for scene in ds.list_scene_refs()],
        "scenes": {},
    }
    for ref in ds.list_scene_refs():
        report = ds.validate_scene_data_flow(ref.id)
        try:
            levels = [_json_ready(level) for level in ds.list_levels(ref.id)]
        except Exception as exc:
            levels = []
            level_error = {"code": "multiscale_invalid", "message": str(exc)}
        else:
            level_error = None
        scene_payload = {
            "ref": _json_ready(ref),
            "metadata": ds.read_scene_metadata(ref.id),
            "levels": levels,
            "data_flow": {
                "warnings": [_json_ready(message) for message in report.warnings],
                "errors": [_json_ready(message) for message in report.errors],
            },
        }
        if level_error is not None:
            scene_payload["level_error"] = level_error
        try:
            scene_payload["ome"] = _json_ready(ds.read_scene_ome_metadata(ref.id))
        except FileNotFoundError:
            scene_payload["ome"] = None
        payload["scenes"][ref.id] = scene_payload
    logger.info("%s", json.dumps(payload, indent=2, default=str))
    return 0


def _cmd_repair(args) -> int:
    logger = setup_logging(args.log_level)
    ds = open_dataset(args.input, mode="a" if args.persist or args.persist_table else "r")
    scene_ids = args.scene or ds.list_scenes()
    payload = {"path": str(ds.path), "scenes": {}}
    for scene_id in scene_ids:
        if args.persist_table:
            _, table_report = ds.ensure_plane_table(scene_id, rebuild=args.rebuild_table)
        else:
            _, table_report = ds.build_plane_table(scene_id, persist=False)
        repair_report = ds.repair_axis_metadata(scene_id, persist=args.persist)
        payload["scenes"][scene_id] = {
            "table": {
                "row_count": table_report.row_count,
                "persisted": table_report.persisted,
                "warnings": [_json_ready(message) for message in table_report.warnings],
            },
            "repair": {
                "persisted": repair_report.persisted,
                "axis_states": {axis: _json_ready(state) for axis, state in repair_report.axis_states.items()},
                "warnings": [_json_ready(message) for message in repair_report.warnings],
                "errors": [_json_ready(message) for message in repair_report.errors],
            },
        }
    logger.info("%s", json.dumps(payload, indent=2, default=str))
    return 0


def _json_ready(value):
    if is_dataclass(value):
        return {field.name: _json_ready(getattr(value, field.name)) for field in fields(value)}
    if isinstance(value, MappingProxyType):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    return value


def main() -> int:
    parser = argparse.ArgumentParser(prog="microio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_inspect = sub.add_parser("inspect", help="Inspect OME-Zarr dataset metadata")
    p_inspect.add_argument("--input", required=True)
    p_inspect.add_argument("--log-level", default="INFO")
    p_inspect.set_defaults(func=_cmd_inspect)

    p_repair = sub.add_parser("repair", help="Validate/repair scene metadata and plane tables")
    p_repair.add_argument("--input", required=True)
    p_repair.add_argument("--scene", action="append")
    p_repair.add_argument("--persist", action="store_true")
    p_repair.add_argument("--persist-table", action="store_true")
    p_repair.add_argument("--rebuild-table", action="store_true")
    p_repair.add_argument("--log-level", default="INFO")
    p_repair.set_defaults(func=_cmd_repair)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
