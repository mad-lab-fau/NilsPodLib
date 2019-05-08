"""Legacy support helper to convert older NilsPod files into new versions."""
import warnings
from distutils.version import StrictVersion
from typing import NoReturn, Tuple
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

    data_bytes = fix_little_endian_counter(data_bytes, packet_size).flatten()

    header = insert_missing_bytes_11_2(header)
    header[4:6] = split_sampling_rate_byte_11_2(header[3])
    header[2] = convert_sensor_enabled_flag_11_2(header[2])

    with open(out_path, 'wb+') as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))


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

    # convert other sensors if enabled
    for old, new in conversion_map.items():
        if bool(byte & old) is True:
            out_byte |= new
    return out_byte


def insert_missing_bytes_11_2(header_bytes):
    header_bytes = np.insert(header_bytes, 3, 0x00)

    header_bytes = np.insert(header_bytes, 46, [0x00] * 2)

    return header_bytes


def split_sampling_rate_byte_11_2(sampling_rate_byte: int) -> Tuple[int, int]:
    return sampling_rate_byte & 0x0F, sampling_rate_byte & 0xF0
