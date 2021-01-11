import numpy as np
import pytest

from nilspodlib.utils import convert_little_endian, validate_existing_overlap


@pytest.mark.parametrize(
    "byte_vals, ints",
    [([0], 0), ([0, 0, 0, 0], 0), ([1, 0], 1), ([0, 1], 256), ([1, 1], 257)],
)
def test_little_endian_simple(byte_vals, ints):
    assert convert_little_endian(byte_vals) == ints


@pytest.mark.parametrize("bytes, ints", [(np.array([[0, 1], [1, 0]]), np.array([256, 1]))])
def test_little_endian_array(bytes, ints):
    assert np.array_equal(convert_little_endian(bytes), ints)


@pytest.mark.parametrize(
    "starts, stops, result",
    [
        ([0, 1], [1, 2], False),
        ([0, 0], [1, 1], True),
        ([0, 0.5], [1, 1.5], True),
        ([0, 0, 0], [1, 1, 1], True),
        ([0, 0, 1], [1, 1, 2], False),
    ],
)
def test_validate_overlap(starts, stops, result):
    assert validate_existing_overlap(np.array(starts), np.array(stops)) == result


@pytest.mark.parametrize(
    "starts, stops",
    [([1, 1], [0, 0]), ([0, 0], [-1, 1])],
)
def test_validate_overlap_error(starts, stops):
    with pytest.raises(ValueError):
        validate_existing_overlap(np.array(starts), np.array(stops))
