from pathlib import Path

from microio.reader.open import open_dataset
from tests.helpers.datasets import cleanup_dataset, create_large_metadata_fixture, fresh_dataset_path


def test_generated_reference_metadata_is_large_but_readable():
    dataset = fresh_dataset_path("metadata_guard")
    try:
        create_large_metadata_fixture(dataset, image_count=300, plane_count=48)
        xml_path = dataset / "OME" / "METADATA.ome.xml"
        txt = xml_path.read_text(encoding="utf-8")
        ds = open_dataset(dataset)

        assert "<OME" in txt
        assert len(txt) > 100000
        assert ds.read_scene_ome_metadata("0").name == "scene_0"
        assert xml_path.stat().st_size < 2 * 1024 * 1024
    finally:
        cleanup_dataset(dataset)
