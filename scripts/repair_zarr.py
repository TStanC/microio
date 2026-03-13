from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from microio import open_dataset
from microio.common.logging_utils import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build plane tables and repair reader-side metadata for a bioformats2raw OME-Zarr dataset."
    )
    parser.add_argument("--input", required=True, help="Path to the OME-Zarr dataset to inspect or repair.")
    parser.add_argument("--scene", action="append", help="Optional scene id to process. Can be provided multiple times.")
    parser.add_argument(
        "--persist-table",
        action="store_true",
        help="Persist the generated plane table under scene/tables/axes_trajectory.",
    )
    parser.add_argument(
        "--rebuild-table",
        action="store_true",
        help="Rebuild an existing plane table instead of reusing a compatible one.",
    )
    parser.add_argument(
        "--persist-repair",
        action="store_true",
        help="Persist safe z/t metadata repairs back into the dataset.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level, for example DEBUG, INFO, WARNING, or ERROR.",
    )
    args = parser.parse_args()

    setup_logging(args.log_level)
    mode = "a" if args.persist_table or args.persist_repair else "r"
    ds = open_dataset(args.input, mode=mode)
    scene_ids = args.scene or ds.list_scenes()

    payload: dict[str, object] = {
        "path": str(ds.path),
        "mode": mode,
        "scenes": {},
    }

    for scene_id in scene_ids:
        if args.persist_table:
            table, table_report = ds.ensure_plane_table(scene_id, rebuild=args.rebuild_table)
        else:
            table, table_report = ds.build_plane_table(scene_id, persist=False)

        repair_report = ds.repair_axis_metadata(scene_id, persist=args.persist_repair)

        payload["scenes"][scene_id] = {
            "table": {
                "row_count": table_report.row_count,
                "persisted": table_report.persisted,
                "columns": sorted(table.keys()),
                "warnings": [message.__dict__ for message in table_report.warnings],
            },
            "repair": {
                "persisted": repair_report.persisted,
                "axis_states": {axis: state.__dict__ for axis, state in repair_report.axis_states.items()},
                "warnings": [message.__dict__ for message in repair_report.warnings],
                "errors": [message.__dict__ for message in repair_report.errors],
            },
        }

    print(json.dumps(payload, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
