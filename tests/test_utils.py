from NilsPodLib.utils import convert_little_endian
import pytest
import numpy as np


@pytest.mark.parametrize('bytes, ints', [
    ([0, ], 0),
    ([0, 0, 0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 256),
    ([1, 1], 257),
])
def test_little_endian_simple(bytes, ints):
    assert convert_little_endian(bytes) == ints


@pytest.mark.parametrize('bytes, ints', [
    (np.array([[0, 1], [1, 0]]), np.array([256, 1]))
])
def test_little_endian_array(bytes, ints):
    assert np.array_equal(convert_little_endian(bytes), ints)

