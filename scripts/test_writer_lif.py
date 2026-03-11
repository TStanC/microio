from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from microio import convert_file, setup_java


JAVA_HOME = r"C:\Users\go24meg\AppData\Local\miniconda3\envs\ImageAB_ultrack_v2\Library\lib\jvm"


def _check_report(report, out: Path, *, expected_total: int | None = None, expected_converted: int | None = None) -> None:
    assert out.exists(), f"Missing output: {out}"
    assert (out / "OME" / "METADATA.ome.xml").exists(), "Missing METADATA.ome.xml"
    assert len(report.scene_reports) > 0, "No scenes processed"
    converted = sum(1 for s in report.scene_reports if s.converted)
    assert any(s.converted for s in report.scene_reports), "No scene converted"
    if expected_total is not None:
        assert len(report.scene_reports) == expected_total, (
            f"Expected {expected_total} scene reports, got {len(report.scene_reports)}"
        )
    if expected_converted is not None:
        assert converted == expected_converted, f"Expected {expected_converted} converted scenes, got {converted}"


def _run_case(label: str, input_path: str, output_path: str, **kwargs):
    print(f"Running {label}: {input_path} -> {output_path}")
    report = convert_file(input_path, output_path, overwrite=True, **kwargs)
    print(
        f"{label}: scenes={len(report.scene_reports)} "
        f"converted={sum(1 for s in report.scene_reports if s.converted)} errors={len(report.errors)}"
    )
    if report.errors:
        print("Errors:", report.errors)
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="data_in/lif/251202_LMNB1_Caax_40x.lif")
    p.add_argument("--output", default="data_out/microio_lif_test.zarr")
    p.add_argument(
        "--scene-selection-input",
        default="data_in/lif/250225_CTN_H2B_pattern_E8Flex.lif",
        help="LIF file used for include/exclude scene-selection regression checks.",
    )
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--include-scene-index", type=int, action="append")
    args = p.parse_args()

    setup_java(fetch_mode="never", java_version="21", java_home=JAVA_HOME)

    report = _run_case(
        "lif-smoke",
        args.input,
        args.output,
        include_scene_index=args.include_scene_index or [0],
        exclude_scene_regex=["Overview", "Preview"],
        max_t=2,
        max_c=2,
        max_z=2,
    )
    _check_report(report, Path(args.output))

    selection_input = Path(args.scene_selection_input)
    if selection_input.exists():
        include_out = Path("data_out/microio_lif_include_one.zarr")
        include_report = _run_case(
            "lif-include-one",
            str(selection_input),
            str(include_out),
            include_scene_name=["E8Flex/Stripes/300_300_Merged"],
            max_t=1,
            max_c=1,
            max_z=1,
        )
        _check_report(include_report, include_out, expected_total=1, expected_converted=1)
        assert [s.scene_index for s in include_report.scene_reports] == [15], "Expected only scene index 15"

        exclude_out = Path("data_out/microio_lif_exclude_climate.zarr")
        exclude_report = _run_case(
            "lif-exclude-climate",
            str(selection_input),
            str(exclude_out),
            exclude_scene_regex=["Climate"],
            max_t=1,
            max_c=1,
            max_z=1,
        )
        _check_report(exclude_report, exclude_out, expected_total=26, expected_converted=26)
        assert all(s.scene_index != 0 for s in exclude_report.scene_reports), "Climate scene should be excluded"

    print("LIF writer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
