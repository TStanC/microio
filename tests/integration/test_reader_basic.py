from pathlib import Path

from microio.reader.open import open_dataset


def test_open_reference_dataset():
    path = Path("data_in/zarr/LMNB1 RFP 60xOHR stack TL 50_tZ-Stack_20251113_1706.zarr")
    ds = open_dataset(path)
    scenes = ds.list_scenes()
    assert scenes
    assert "OME" not in scenes
    assert isinstance(ds.read_root_metadata(), dict)
