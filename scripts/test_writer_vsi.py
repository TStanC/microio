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
    p.add_argument("--input", default="data_in/vsi/LMNB1_undiff_60x_002.vsi")
    p.add_argument("--output", default="data_out/microio_vsi_test.zarr")
    p.add_argument("--overwrite", action="store_true")
    args = p.parse_args()

    setup_java(fetch_mode="never", java_version="21", java_home=JAVA_HOME)

    report = _run_case(
        "vsi-smoke",
        args.input,
        args.output,
        exclude_scene_regex=["macro", "Macro", "Overview", "Preview"],
        max_t=2,
        max_c=1,
        max_z=4,
    )
    _check_report(report, Path(args.output), expected_total=1, expected_converted=1)
    assert [s.scene_id for s in report.scene_reports] == ["C555"], "Expected only the non-macro scene"

    exclude_out = Path("data_out/microio_vsi_exclude_macro.zarr")
    exclude_report = _run_case(
        "vsi-exclude-macro",
        args.input,
        str(exclude_out),
        exclude_scene_regex=["macro", "Macro"],
        max_t=2,
        max_c=1,
        max_z=4,
    )
    _check_report(exclude_report, exclude_out, expected_total=1, expected_converted=1)
    assert [s.scene_id for s in exclude_report.scene_reports] == ["C555"], "Expected only the non-macro scene"

    include_out = Path("data_out/microio_vsi_include_macro.zarr")
    include_report = _run_case(
        "vsi-include-macro",
        args.input,
        str(include_out),
        include_scene_name=["macro image"],
        max_t=2,
        max_c=1,
        max_z=4,
    )
    assert len(include_report.scene_reports) == 1, "Expected exactly one macro scene report"
    assert [s.scene_id for s in include_report.scene_reports] == ["macro_image"], "Expected the macro scene"
    assert include_report.errors == ["scene=macro_image: axes length (5) must match number of dimensions (6)"], (
        "Macro include should currently fail only at the downstream 6D writer step"
    )

    print("VSI writer tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
