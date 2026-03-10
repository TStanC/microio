from pathlib import Path
import shutil
import uuid

import numpy as np
import zarr

from microio.reader.open import open_dataset


def test_build_axis_positions_from_order():
    tmp_path = Path("data_out") / f"_test_table_reader_{uuid.uuid4().hex}.zarr"
    if tmp_path.exists():
        shutil.rmtree(tmp_path)
    root = zarr.open(tmp_path, mode="w")
    scene = root.create_group("scene0")
    scene.attrs["microio"] = {
        "axis_resolution": {
            "t": {"value": 2.5},
            "z": {"value": 0.75},
            "c": {"value": 1.0},
            "y": {"value": 1.0},
            "x": {"value": 1.0},
        }
    }
    tables = scene.create_group("tables")
    table = tables.create_group("axes_trajectory")
    table.create_array("the_t", data=np.asarray([0, 0, 1, 1], dtype=np.int32))
    table.create_array("the_c", data=np.asarray([0, 1, 0, 1], dtype=np.int32))
    table.create_array("the_z", data=np.asarray([0, 0, 0, 0], dtype=np.int32))
    table.create_array("positioners_t", data=np.asarray([np.nan, np.nan, np.nan, np.nan], dtype=np.float64))
    table.attrs["axis_metadata"] = {
        "t": {"unit": "abstract", "provenance": "Plane.TheT", "confidence": "medium"},
    }

    try:
        ds = open_dataset(tmp_path)
        built = ds.build_axis_positions("scene0", "t")
        assert built.tolist() == [0.0, 0.0, 2.5, 2.5]
    finally:
        if tmp_path.exists():
            shutil.rmtree(tmp_path)
