from __future__ import annotations

import argparse
from pprint import pprint

from microio import open_dataset


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    args = p.parse_args()

    ds = open_dataset(args.input)
    pprint(ds.read_root_metadata())
    for ref in ds.list_scene_refs():
        print("\\nScene:", ref.id, "index=", ref.index, "name=", ref.name)
        md = ds.read_scene_metadata(ref.id)
        pprint({k: md[k] for k in md.keys() if k in ("ome", "microio", "bioformats2raw.layout")})
        try:
            pprint([level.__dict__ for level in ds.list_levels(ref.id)])
        except Exception as exc:
            print("Level validation error:", exc)
        try:
            tbl = ds.load_plane_table(ref.id, "axes_trajectory")
            print("axes_trajectory columns:", list(tbl.keys()))
        except Exception as exc:
            print("No axes_trajectory table:", exc)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
