from __future__ import annotations

import dask.array as da
import numpy as np

from microio.writer.ngff import _sample_data


def test_sample_data_returns_bounded_numpy_sample_from_dask():
    data = da.from_array(np.arange(2 * 3 * 4 * 32 * 32, dtype=np.uint16).reshape(2, 3, 4, 32, 32), chunks=(1, 1, 2, 16, 16))

    sample = _sample_data(data, target_values=128)

    assert isinstance(sample, np.ndarray)
    assert sample.shape[1] == data.shape[1]
    assert sample.size <= data.size
    assert sample.size < data.size


def test_sample_data_keeps_small_arrays_eager_without_growth():
    data = np.arange(1 * 1 * 2 * 4 * 4, dtype=np.uint16).reshape(1, 1, 2, 4, 4)

    sample = _sample_data(data, target_values=1024)

    assert isinstance(sample, np.ndarray)
    assert sample.shape == data.shape
    assert np.array_equal(sample, data)
