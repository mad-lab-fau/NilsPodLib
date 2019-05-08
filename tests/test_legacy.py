import pytest

from NilsPodLib.header import Header
from NilsPodLib.legacy import fix_little_endian_counter, convert_sensor_enabled_flag_11_2
from NilsPodLib.utils import get_sample_size_from_header_bytes, get_header_and_data_bytes, convert_little_endian
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


def test_sensor_enabled_flag_conversion(test_session_11_2):
    _, header, _ = test_session_11_2
    sensors = convert_sensor_enabled_flag_11_2(header[2])
    enabled_sensors = list()
    for para, val in Header._SENSOR_FLAGS.items():
        if bool(sensors & val) is True:
            enabled_sensors.append(para)

    assert enabled_sensors == ['gyro', 'acc']
