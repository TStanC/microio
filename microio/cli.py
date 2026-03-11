"""Command-line interface for conversion and dataset inspection."""

from __future__ import annotations

import argparse
import json

from microio import convert_file, open_dataset, setup_java
from microio.common.logging_utils import setup_logging


def _cmd_convert(args) -> int:
    """Run conversion command and print structured JSON report."""
    setup_logging(args.log_level)
    setup_java(
        fetch_mode=args.java_fetch_mode,
        java_version=args.java_version,
        java_home=args.java_home,
        logback_path=args.logback,
    )
    report = convert_file(
        args.input,
        args.output,
        target_ngff=args.target_ngff,
        include_scene_index=args.include_scene_index,
        include_scene_name=args.include_scene_name,
        exclude_scene_regex=args.exclude_scene_regex,
        overwrite=args.overwrite,
        chunk_target_mb=args.chunk_target_mb,
        max_workers=args.max_workers,
        max_t=args.max_t,
        max_c=args.max_c,
        max_z=args.max_z,
    )
    print(
        json.dumps(
            {
                "input": str(report.input_path),
                "output": str(report.output_path),
                "target_ngff": report.target_ngff,
                "scenes": [
                    {
                        "scene_index": s.scene_index,
                        "scene_id": s.scene_id,
                        "converted": s.converted,
                        "warnings": s.warnings,
                    }
                    for s in report.scene_reports
                ],
                "errors": report.errors,
                "warnings": report.warnings,
                "fallback_events": report.fallback_events,
            },
            indent=2,
        )
    )
    return 0 if not report.errors else 1


def _cmd_inspect(args) -> int:
    """Run inspection command and print root/scene metadata JSON."""
    setup_logging(args.log_level)
    ds = open_dataset(args.input)
    scenes = ds.list_scenes()
    payload = {
        "path": str(ds.path),
        "root": ds.read_root_metadata(),
        "scenes": {},
    }
    for s in scenes:
        payload["scenes"][s] = {
            "metadata": ds.read_scene_metadata(s),
            "microio": ds.read_microio_extras(s),
        }
    print(json.dumps(payload, indent=2, default=str))
    return 0


def main() -> int:
    """Parse CLI args and dispatch subcommands."""
    parser = argparse.ArgumentParser(prog="microio")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_convert = sub.add_parser("convert", help="Convert VSI/LIF to OME-Zarr")
    p_convert.add_argument("--input", required=True)
    p_convert.add_argument("--output", required=True)
    p_convert.add_argument("--target-ngff", default="0.5")
    p_convert.add_argument("--include-scene-index", type=int, action="append")
    p_convert.add_argument("--include-scene-name", action="append")
    p_convert.add_argument("--exclude-scene-regex", action="append")
    p_convert.add_argument("--overwrite", action="store_true")
    p_convert.add_argument("--chunk-target-mb", type=int, default=8)
    p_convert.add_argument("--max-workers", type=int, default=1)
    p_convert.add_argument("--max-t", type=int)
    p_convert.add_argument("--max-c", type=int)
    p_convert.add_argument("--max-z", type=int)
    p_convert.add_argument("--java-home")
    p_convert.add_argument("--java-version", default="21")
    p_convert.add_argument("--java-fetch-mode", default="never")
    p_convert.add_argument("--logback")
    p_convert.add_argument("--log-level", default="INFO")
    p_convert.set_defaults(func=_cmd_convert)

    p_inspect = sub.add_parser("inspect", help="Inspect OME-Zarr dataset")
    p_inspect.add_argument("--input", required=True)
    p_inspect.add_argument("--log-level", default="INFO")
    p_inspect.set_defaults(func=_cmd_inspect)

    args = parser.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
