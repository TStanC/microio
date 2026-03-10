from __future__ import annotations

import argparse
from pathlib import Path

from microio import convert_file, setup_java


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_in/lif/251202_LMNB1_Caax_40x.lif")
    p.add_argument("--output", default="data_out/microio_lif_test.zarr")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-scene-index", type=int, action="append")
    args = p.parse_args()

    setup_java(fetch_mode="never", java_version="21")
    report = convert_file(
        args.input,
        args.output,
        overwrite=args.overwrite,
        include_scene_index=args.include_scene_index or [0],
        exclude_scene_regex=["Overview", "Preview"],
        max_t=2,
        max_c=2,
        max_z=2,
    )

    out = Path(args.output)
    assert out.exists(), f"Missing output: {out}"
    assert (out / "OME" / "METADATA.ome.xml").exists(), "Missing METADATA.ome.xml"
    assert len(report.scene_reports) > 0, "No scenes processed"
    assert any(s.converted for s in report.scene_reports), "No scene converted"
    print("LIF writer test passed", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
