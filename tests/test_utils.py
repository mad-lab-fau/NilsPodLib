from NilsPodLib.utils import convert_little_endian
import pytest


@pytest.mark.parametrize('bytes, ints', [
    ([0, ], 0),
    ([0, 0, 0, 0], 0),
    ([1, 0], 1),
    ([0, 1], 256),
    ([1, 1], 257)
])
def test_little_endian(bytes, ints):
    assert convert_little_endian(bytes) == ints
