"""Legacy support helper to convert older NilsPod files into new versions."""
import warnings
from distutils.version import StrictVersion
from typing import Tuple, Callable, Optional, Union
import numpy as np

from NilsPodLib.utils import path_t, get_header_and_data_bytes, get_strict_version_from_header_bytes, \
    get_sample_size_from_header_bytes

CONVERSION_DICT = {
    '12_0': {'min': StrictVersion('0.11.255'),
             'max': StrictVersion('0.13.255')},
    '11_2': {'min': StrictVersion('0.11.2'),
             'max': StrictVersion('0.11.255')}
}


def find_conversion_function(version: StrictVersion, in_memory: Optional[bool] = True,
                             return_name: Optional[bool] = False) -> Union[Callable, str]:
    """Return a suitable conversion funtion for a specific legacy version.

    This will either return one of the `load_{}` functions, if `in_memory` is True or the `convert_{}` variant if False
    """
    # TODO: test
    for k, v in CONVERSION_DICT.items():
        if v['min'] <= version < v['max']:
            n = 'load_' if in_memory else 'convert_'
            if return_name:
                return n + k
            return globals()[n + k]
    raise VersionError('No suitable conversion function found for {}'.format(version))


def convert_12_0(in_path: path_t, out_path: path_t) -> None:
    """Convert a session recorded with a firmware version >0.11.255 and <0.13.255 to the most up-to-date format.

    This will update the firmware version to 0.13.255 to identify converted sessions.
    Potential version checks in the library will use <0.13.255 to check for compatibility.

    Warnings:
        After the update the following features will not work:
            - The sync group was removed and hence can not be read anymore

    Args:
        in_path: path to 0.12.x / 0.13.x file
        out_path: path to converted 0.13.255 file
    """
    header, data_bytes = get_header_and_data_bytes(in_path)
    header, data_bytes = load_12_0(header, data_bytes)

    with open(out_path, 'wb+') as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))


def load_12_0(header: np.ndarray, data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session recorded with a firmware version >0.11.255 and <0.13.255 to the most up-to-date format.

    This will update the firmware version to 0.13.255 to identify converted sessions.
    Potential version checks in the library will use <0.13.255 to check for compatibility.

    Warnings:
        After the update the following features will not work:
            - The sync group was removed and hence can not be read anymore

    Args:
        header: bytes containing all the legacy header information
        data_bytes: raw bytes representing the data
    """
    min_v = CONVERSION_DICT['12_0']['min']
    max_v = CONVERSION_DICT['12_0']['max']
    version = get_strict_version_from_header_bytes(header)

    if not (min_v <= version < max_v):
        raise VersionError('This converter is meant for files recorded with Firmware version after {} and before {}'
                           ' not {}'.format(min_v, max_v, version))

    header = shift_bytes_12_0(header)

    # Update firmware version
    header[-2] = 13
    header[-1] = 255

    return header, data_bytes


def convert_11_2(in_path: path_t, out_path: path_t) -> None:
    """Convert a session recorded with a 0.11.<2 firmware to the most up-to-date format.

    This will update the firmware version to 0.13.255 to identify converted sessions.
    Potential version checks in the library will use <0.13.255 to check for compatibility.

    Warnings:
        After the update the following features will not work:
            - The battery sensor does not exist anymore and hence, is not supported in the converted files
            - The sync group was removed and hence can not be read anymore

    Args:
        in_path: path to 0.11.2 file
        out_path: path to converted 0.13.255 file
    """
    header, data_bytes = get_header_and_data_bytes(in_path)
    header, data_bytes = load_11_2(header, data_bytes)

    with open(out_path, 'wb+') as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))


def load_11_2(header: np.ndarray, data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session recorded with a 0.11.<2 firmware to the most up-to-date format.

    This will update the firmware version to 0.13.255 to identify converted sessions.
    Potential version checks in the library will use <0.13.255 to check for compatibility.

    Warnings:
        After the update the following features will not work:
            - The battery sensor does not exist anymore and hence, is not supported in the converted files
            - The sync group was removed and hence can not be read anymore

    Args:
        header: bytes containing all the legacy header information
        data_bytes: raw bytes representing the data
    """
    min_v = CONVERSION_DICT['11_2']['min']
    max_v = CONVERSION_DICT['11_2']['max']
    version = get_strict_version_from_header_bytes(header)

    if not (min_v <= version < max_v):
        raise VersionError('This converter is meant for files recorded with Firmware version after {} and before {}'
                           ' not {}'.format(min_v, max_v, version))

    packet_size = get_sample_size_from_header_bytes(header)

    data_bytes = fix_little_endian_counter(data_bytes, packet_size).flatten()

    header = insert_missing_bytes_11_2(header)
    header[3:5] = split_sampling_rate_byte_11_2(header[3])
    header[2] = convert_sensor_enabled_flag_11_2(header[2])

    # adapt to new header size:
    header[0] = len(header)

    # Update firmware version
    header[-1] = 255

    header, data_bytes = load_12_0(header, data_bytes)
    return header, data_bytes


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
        0x08: 0x80  # temperature
    }

    # always enable acc for old sessions:
    out_byte = 0x01

    # convert other sensors if enabled
    for old, new in conversion_map.items():
        if bool(byte & old) is True:
            out_byte |= new
    return out_byte


def insert_missing_bytes_11_2(header_bytes):
    header_bytes = np.insert(header_bytes, 4, 0x00)

    header_bytes = np.insert(header_bytes, 47, [0x00] * 2)

    return header_bytes


def shift_bytes_12_0(header_bytes):
    # remove old sync_group byte:
    header_bytes = np.delete(header_bytes, 7)

    # Add new empty byte after enabled sensors
    header_bytes = np.insert(header_bytes, 3, 0x00)

    return header_bytes


def split_sampling_rate_byte_11_2(sampling_rate_byte: int) -> Tuple[int, int]:
    return sampling_rate_byte & 0x0F, sampling_rate_byte & 0xF0


def legacy_support_check(version: StrictVersion, as_warning: bool = False):
    if version < StrictVersion('0.11.2'):
        msg = 'You are using a version ({}) previous to 0.11.2. This version is not supported!'.format(version)
    elif version >= StrictVersion('0.13.255'):
        return
    else:
        try:
            converter = find_conversion_function(version, in_memory=False, return_name=True)
            msg = 'You are using a version ({}) which is only supported by legacy support.' \
                  ' Use `{}` to update the binary format to a newer version' \
                  ' or use `legacy_support="resolve"` when loading the file'.format(version, converter)
        except VersionError:
            msg = 'You are using a version completely unknown version: {}'.format(version)

    if as_warning is True:
        warnings.warn(msg)
    else:
        raise VersionError(msg)


class VersionError(Exception):
    pass
