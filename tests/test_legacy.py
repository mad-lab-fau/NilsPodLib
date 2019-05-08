import pytest

from NilsPodLib.legacy import fix_little_endian_counter
from NilsPodLib.utils import get_sample_size_from_header_bytes, get_header_and_data_bytes, convert_little_endian, \
    read_binary_uint8
from tests.conftest import TEST_LEGACY_DATA
import numpy as np


@pytest.fixture()
def test_session_11_2():
    path = TEST_LEGACY_DATA / 'NilsPodX-8433_20190412_172203.bin'
    header, data_bytes = get_header_and_data_bytes(path)
    return path, header, data_bytes


def test_endian_conversion(test_session_11_2):
    _, header, data = test_session_11_2
    sample_size = get_sample_size_from_header_bytes(header)
    data = fix_little_endian_counter(data, sample_size)
    counter_after = convert_little_endian(np.atleast_2d(data[:, -4:]).T, dtype=float)
    assert np.all(np.diff(counter_after) == 1.)

