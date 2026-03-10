from pathlib import Path
import json


def test_reference_scene_metadata_is_large_but_readable():
    path = Path("data_in/zarr/LMNB1 RFP 60xOHR stack TL 50_tZ-Stack_20251113_1706.zarr/c555/zarr.json")
    txt = path.read_text(encoding="utf-8")
    data = json.loads(txt)
    assert "attributes" in data
    assert len(txt) > 100000
