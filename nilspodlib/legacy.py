"""Legacy support helper to convert older NilsPod files into new versions."""
import warnings
from distutils.version import StrictVersion
from typing import Tuple, Callable, Optional, Union

import numpy as np

from nilspodlib.exceptions import LegacyWarning, VersionError, CorruptedPackageWarning
from nilspodlib.consts import SENSOR_SAMPLE_LENGTH
from nilspodlib.utils import (
    path_t,
    get_header_and_data_bytes,
    get_strict_version_from_header_bytes,
    get_sample_size_from_header_bytes,
)

CONVERSION_DICT = {
    "18_0": {"min": StrictVersion("0.13.255"), "max": StrictVersion("0.17.255")},
    "12_0": {"min": StrictVersion("0.11.255"), "max": StrictVersion("0.13.255")},
    "11_2": {"min": StrictVersion("0.11.2"), "max": StrictVersion("0.11.255")},
}

MIN_NON_LEGACY_VERSION = StrictVersion("0.18.0")


def find_conversion_function(
    version: StrictVersion, in_memory: Optional[bool] = True, return_name: Optional[bool] = False
) -> Union[Callable, str]:
    """Find a method that is able to convert a recording from one version to the other.

    This will either return one of the `load_{}` functions, if `in_memory` is True or the `convert_{}` variant if False.
    """
    if version >= MIN_NON_LEGACY_VERSION:
        return lambda x, y: (x, y)

    for k, v in CONVERSION_DICT.items():
        if v["min"] <= version < v["max"]:
            n = "load_" if in_memory else "convert_"
            if return_name:
                return n + k
            return globals()[n + k]
    raise VersionError("No suitable conversion function found for {}".format(version))

def convert_18_0(in_path: path_t, out_path: path_t) -> None:
    """Convert a session recorded with a firmware version >0.13.255 and <0.17.255 to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    Parameters
    ----------
    in_path :
        path to 0.14.x - 0.17.x file
    out_path :
        path to converted 0.17.255 file

    """
    header, data_bytes = get_header_and_data_bytes(in_path)
    header, data_bytes = load_18_0(header, data_bytes)

    with open(out_path, "wb+") as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))

def load_18_0(header: np.ndarray, data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session recorded with a firmware version >0.13.255 and <0.17.255 to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    Parameters
    ----------
    header :
        bytes containing all the legacy header information
    data_bytes :
        raw bytes representing the data

    """
    min_v = CONVERSION_DICT["18_0"]["min"]
    max_v = CONVERSION_DICT["18_0"]["max"]
    version = get_strict_version_from_header_bytes(header)

    if not min_v <= version < max_v:
        raise VersionError(
            "This converter is meant for files recorded with Firmware version after {} and before {}"
            " not {}".format(min_v, max_v, version)
        )

    analog_enabled = header[2] & 0x10
    if analog_enabled:
        # convert analog channels from uint8 to uint16
        data_bytes = _convert_analog_uint8_to_uint16_18_0(data_bytes, header)
        # Update sample size
        header[1] = header[1] + 3

    # Update firmware version
    header[-2] = 17
    header[-1] = 255

    return header, data_bytes


def convert_12_0(in_path: path_t, out_path: path_t) -> None:
    """Convert a session recorded with a firmware version >0.11.255 and <0.13.255 to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    .. warning:: After the update the following features will not work:
                    - The sync group was removed and hence can not be read anymore

    Parameters
    ----------
    in_path :
        path to 0.12.x / 0.13.x file
    out_path :
        path to converted 0.17.255 file

    """
    header, data_bytes = get_header_and_data_bytes(in_path)
    header, data_bytes = load_12_0(header, data_bytes)

    with open(out_path, "wb+") as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))


def load_12_0(header: np.ndarray, data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session recorded with a firmware version >0.11.255 and <0.13.255 to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    .. warning:: After the update the following features will not work:
                    - The sync group was removed and hence can not be read anymore

    Parameters
    ----------
    header :
        bytes containing all the legacy header information
    data_bytes :
        raw bytes representing the data

    """
    min_v = CONVERSION_DICT["12_0"]["min"]
    max_v = CONVERSION_DICT["12_0"]["max"]
    version = get_strict_version_from_header_bytes(header)

    if not min_v <= version < max_v:
        raise VersionError(
            "This converter is meant for files recorded with Firmware version after {} and before {}"
            " not {}".format(min_v, max_v, version)
        )

    header = _shift_bytes_12_0(header)

    # stack conversion functions
    header, data_bytes = load_18_0(header, data_bytes)

    return header, data_bytes


def convert_11_2(in_path: path_t, out_path: path_t) -> None:
    """Convert a session recorded with a 0.11.<2 firmware to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    .. warning:: After the update the following features will not work:
                    - The battery sensor_type does not exist anymore and hence, is not supported in the converted files
                    - The sync group was removed and hence can not be read anymore

    Parameters
    ----------
    in_path :
        path to 0.11.2 file
    out_path :
        path to converted 0.17.255 file

    """
    header, data_bytes = get_header_and_data_bytes(in_path)
    header, data_bytes = load_11_2(header, data_bytes)

    with open(out_path, "wb+") as f:
        f.write(bytearray(header))
        f.write(bytearray(data_bytes))


def load_11_2(header: np.ndarray, data_bytes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Convert a session recorded with a 0.11.<2 firmware to the most up-to-date format.

    This will update the firmware version to 0.17.255 to identify converted sessions.
    Potential version checks in the library will use <0.17.255 to check for compatibility.

    .. warning:: After the update the following features will not work:
                    - The battery sensor_type does not exist anymore and hence, is not supported in the converted files
                    - The sync group was removed and hence can not be read anymore

    Parameters
    ----------
    header :
        Bytes containing all the legacy header information
    data_bytes :
        Raw bytes representing the data

    """
    min_v = CONVERSION_DICT["11_2"]["min"]
    max_v = CONVERSION_DICT["11_2"]["max"]
    version = get_strict_version_from_header_bytes(header)

    if not min_v <= version < max_v:
        raise VersionError(
            "This converter is meant for files recorded with Firmware version after {} and before {}"
            " not {}".format(min_v, max_v, version)
        )

    packet_size = get_sample_size_from_header_bytes(header)

    data_bytes = _fix_little_endian_counter(data_bytes, packet_size).flatten()

    header = _insert_missing_bytes_11_2(header)
    header[3:5] = _split_sampling_rate_byte_11_2(header[3])
    header[2] = _convert_sensor_enabled_flag_11_2(header[2])

    # adapt to new header size:
    header[0] = len(header)

    # stack conversion functionss
    header, data_bytes = load_12_0(header, data_bytes)
    header, data_bytes = load_18_0(header, data_bytes)

    return header, data_bytes


def _convert_analog_uint8_to_uint16_18_0(data_bytes, header_bytes):
    """Convert the data format of analog channels of uint8 to uint16."""
    old_sample_size = header_bytes[1]
    temp_enabled = header_bytes[2] & 0x80

    if(len(data_bytes) % old_sample_size):
        data_bytes = data_bytes[:(len(data_bytes) // old_sample_size)*old_sample_size]
        warnings.warn("Number of bytes in binary data does not match sample size indicated by header. "
                      "This can be caused by a bug affecting all synchronised sessions recorded with firmware versions "
                      "before 0.14.0. ", CorruptedPackageWarning)

    data_bytes = data_bytes.reshape(len(data_bytes) // old_sample_size, old_sample_size)

    # build new array to hold new data format
    data_bytes_converted = np.zeros((data_bytes.shape[0], old_sample_size + 3))

    offset = SENSOR_SAMPLE_LENGTH['counter'][0]

    if temp_enabled:
        offset = SENSOR_SAMPLE_LENGTH['counter'][0] + SENSOR_SAMPLE_LENGTH['temperature'][0]

    data_bytes_converted[:, -offset:] = data_bytes[:, -offset:]
    data_bytes_converted[:, -offset - 2] = data_bytes[:, -offset - 1]
    data_bytes_converted[:, -offset - 4] = data_bytes[:, -offset - 2]
    data_bytes_converted[:, -offset - 6] = data_bytes[:, -offset - 3]

    remaining_bytes = old_sample_size - 3 - offset

    data_bytes_converted[:, :remaining_bytes] = data_bytes[:, :remaining_bytes]

    return data_bytes_converted.flatten()


def _fix_little_endian_counter(data_bytes, packet_size):
    """Convert the big endian counter of older firmware versions to a small endian counter."""
    expected_samples = len(data_bytes) // packet_size
    data_bytes = data_bytes[: expected_samples * packet_size]
    data = np.reshape(data_bytes, (expected_samples, int(packet_size)))
    data[:, -4:] = np.flip(data[:, -4:], axis=-1)
    return data


def _convert_sensor_enabled_flag_11_2(byte):
    """Convert the old sensor enabled flags to the new one."""
    conversion_map = {0x01: 0x02, 0x02: 0x10, 0x04: 0x08, 0x08: 0x80}  # gyro  # analog  # baro  # temperature

    # always enable acc for old sessions:
    out_byte = 0x01

    # convert other sensors if enabled
    for old, new in conversion_map.items():
        if bool(byte & old) is True:
            out_byte |= new
    return out_byte


def _insert_missing_bytes_11_2(header_bytes):
    """Insert header bytes that were added after 11.2."""
    header_bytes = np.insert(header_bytes, 4, 0x00)

    header_bytes = np.insert(header_bytes, 47, [0x00] * 2)

    return header_bytes


def _shift_bytes_12_0(header_bytes):
    """Move old bytestructure, as it was changed for versions after 12.0."""
    # remove old sync_group byte:
    header_bytes = np.delete(header_bytes, 7)

    # Add new empty byte after enabled sensors
    header_bytes = np.insert(header_bytes, 3, 0x00)

    return header_bytes


def _split_sampling_rate_byte_11_2(sampling_rate_byte: int) -> Tuple[int, int]:
    """Separate sampling rate into its own byte."""
    return sampling_rate_byte & 0x0F, sampling_rate_byte & 0xF0


def legacy_support_check(version: StrictVersion, as_warning: bool = False):
    """Check if a file recorded with a specific fileformat version can be converted using legacy support.

    Parameters
    ----------
    version :
        The version to check for.
    as_warning :
        If True only a Warning instead of an error is raised, if legacy support is required for the dataset.

    """
    if version < StrictVersion("0.11.2"):
        msg = "You are using a version ({}) previous to 0.11.2. This version is not supported!".format(version)
    elif version >= StrictVersion("0.13.255"):
        return
    else:
        try:
            converter = find_conversion_function(version, in_memory=False, return_name=True)
            msg = (
                "You are using a version ({}) which is only supported by legacy support."
                " Use `{}` to update the binary format to a newer version"
                ' or use `legacy_support="resolve"` when loading the file'.format(version, converter)
            )
        except VersionError:
            msg = "You are using a version completely unknown version: {}".format(version)

    if as_warning is True:
        warnings.warn(msg, LegacyWarning)
    else:
        raise VersionError(msg)
