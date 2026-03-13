from microio.common.units import normalize_unit


def test_normalize_known():
    assert normalize_unit("um")[0] == "micrometer"
    assert normalize_unit("s")[0] == "second"


def test_normalize_corrupted():
    norm, warn = normalize_unit("痠")
    assert norm == "micrometer"
    assert warn == "unit_corrupted_coerced"


def test_normalize_corrupted_millisecond():
    norm, warn = normalize_unit("10^-3s^1")
    assert norm == "millisecond"
    assert warn == "unit_corrupted_coerced"


def test_normalize_unknown():
    norm, warn = normalize_unit("nonsense_unit")
    assert norm == "unknown"
    assert warn == "unit_unknown"
