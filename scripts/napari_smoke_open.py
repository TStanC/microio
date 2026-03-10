from __future__ import annotations

import argparse


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    import napari

    viewer = napari.Viewer()
    viewer.open(args.input, plugin="napari-ome-zarr")
    print("Opened in napari:", args.input)
    napari.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
