from microio.writer.convert import _unique_scene_id


def test_scene_id_dedup():
    seen = {}
    assert _unique_scene_id("A/B", seen) == "A_B"
    assert _unique_scene_id("A/B", seen) == "A_B__1"
