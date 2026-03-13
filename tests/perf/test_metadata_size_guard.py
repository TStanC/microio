from pathlib import Path


def test_reference_scene_metadata_is_large_but_readable():
    path = (
        Path(__file__).resolve().parents[3]
        / "data_in"
        / "zarr"
        / "lif_test.zarr"
        / "OME"
        / "METADATA.ome.xml"
    )
    txt = path.read_text(encoding="utf-8")
    assert "<OME" in txt
    assert len(txt) > 100000
