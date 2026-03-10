from __future__ import annotations

import argparse
from pathlib import Path

import zarr


def scene_keys(root):
    return sorted([k for k, _ in root.groups() if k != "OME"])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--produced", required=True)
    p.add_argument("--reference", required=True)
    args = p.parse_args()

    prod = zarr.open(Path(args.produced), mode="r")
    ref = zarr.open(Path(args.reference), mode="r")

    print("Produced scenes:", scene_keys(prod))
    print("Reference scenes:", scene_keys(ref))

    print("Produced root ome:", prod.attrs.asdict().get("ome"))
    print("Reference root ome:", ref.attrs.asdict().get("ome"))

    for s in scene_keys(prod):
        g = prod[s]
        md = g.attrs.asdict()
        print(f"Scene={s} has tables?", "tables" in g)
        print(f"Scene={s} has microio extras?", "microio" in md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
