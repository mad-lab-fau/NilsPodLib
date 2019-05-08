"""Legacy support helper to convert older NilsPod files into new versions."""
import warnings
from distutils.version import StrictVersion
from typing import NoReturn
import numpy as np

from NilsPodLib.utils import path_t, get_header_and_data_bytes, get_strict_version_from_header_bytes, \
    get_sample_size_from_header_bytes


def convert_11_2(in_path: path_t, out_path: path_t) -> NoReturn:
    header, data_bytes = get_header_and_data_bytes(in_path)
    version = get_strict_version_from_header_bytes(header)

    if not (StrictVersion('0.11.2') <= version < StrictVersion('0.12.0')):
        warnings.warn('This converter is meant for files recorded with Firmware version after 0.11.2 and before 0.12.0'
                      ' not{}'.format(version))

    packet_size = get_sample_size_from_header_bytes(header)



def fix_little_endian_counter(data_bytes, packet_size):
    expected_samples = len(data_bytes) // packet_size
    data_bytes = data_bytes[:expected_samples * packet_size]
    data = np.reshape(data_bytes, (expected_samples, int(packet_size)))
    data[:, -4:] = np.flip(data[:, -4:], axis=-1)
    return data


def convert_sensor_enabled_flag_11_2(byte):
    conversion_map = {
        0x01: 0x02,  # gyro
        0x02: 0x10,  # analog
        0x04: 0x08,  # baro
        0x08: 0x80   # battery
    }

    # always enable acc for old sessions:
    out_byte = 0x01
    for old, new in conversion_map.items():
        if bool(byte & old) is True:
            out_byte |= new
    return out_byte
