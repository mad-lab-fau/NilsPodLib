#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import struct
import warnings
from distutils.version import StrictVersion
from typing import TypeVar, Union, TYPE_CHECKING, Tuple

import numpy as np
from pathlib import Path

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')


def convert_little_endian(byte_list, dtype=int):
    byte_list = np.array(byte_list).astype(np.uint32)
    number = byte_list[0]
    for i, v in enumerate(byte_list[1:]):
        number |= v << int(8 * (i + 1))
    return number.astype(dtype)


def read_binary_uint8(data_bytes, packet_size, expected_samples):
    packet_size = int(packet_size)
    expected_length = expected_samples * packet_size
    page_size = 2048
    if abs(len(data_bytes) - expected_length) > page_size // packet_size:
        warnings.warn('The provided binary file contains more or less than {0} packages than indicated by the header'
                      ' ({1} vs. {2}). This can be caused by a bug affecting all synchronised sessions recorded with'
                      ' firmware versions before 0.14.0. \n'
                      'The full file will be read to avoid data loss, but this might add up to {0} corrupted packages'
                      ' at the end of the datastream.'.format(page_size // packet_size, expected_samples,
                                                              len(data_bytes) // packet_size))
        expected_length = (len(data_bytes) // packet_size) * packet_size

    elif expected_length > len(data_bytes):
        warnings.warn('The provided binary file contains less samples than indicated by the header.'
                      ' This might mean that the file was corrupted.')
        expected_length = (len(data_bytes) // packet_size) * packet_size

    data_bytes = data_bytes[:expected_length]
    data = np.reshape(data_bytes, (expected_length // packet_size, packet_size))
    return data


def get_header_and_data_bytes(path: path_t) -> Tuple[np.ndarray, np.ndarray]:
    with open(path, 'rb') as f:
        header = f.read(1)
        header_size = header[0]
        header += f.read(header_size - 1)
        data_bytes = np.fromfile(f, dtype=np.dtype('B'))

    header = bytearray(header)
    header_bytes = np.asarray(struct.unpack(str(header_size) + 'b', header[0:header_size]), dtype=np.uint8)

    return header_bytes, data_bytes


def get_sample_size_from_header_bytes(header_bytes: np.ndarray) -> int:
    return int(header_bytes[1])


def get_strict_version_from_header_bytes(header_bytes: np.ndarray) -> StrictVersion:
    return StrictVersion('{}.{}.{}'.format(*(int(x) for x in header_bytes[-3:])))


def inplace_or_copy(obj: T, inplace: bool) -> T:
    if inplace is True:
        return obj
    return copy.deepcopy(obj)


class InvalidInputFileError(Exception):
    pass


class RepeatedCalibrationError(Exception):
    MESSAGE = 'The sensor "{}" is already calibrated. Repeated calibration will lead to wrong values.'

    def __init__(self, sensor_name):
        message = self.MESSAGE.format(sensor_name)
        super().__init__(message)


def datastream_does_not_exist_warning(sensor_name, operation):
    message = 'The datastream "{}" does not exist for the current session.\
     The performed operation "{}" will have not effect'.format(sensor_name, operation)
    return warnings.warn(message)


def load_and_check_cal_info(calibration: Union['CalibrationInfo', path_t]) -> 'CalibrationInfo':
    from imucal import CalibrationInfo  # noqa: F811
    if isinstance(calibration, (Path, str)):
        calibration = CalibrationInfo.from_json_file(calibration)
    if not isinstance(calibration, CalibrationInfo):
        raise ValueError('No valid CalibrationInfo object provided')
    return calibration


def validate_existing_overlap(start_vals: np.ndarray, end_vals: np.ndarray) -> bool:
    if not all(i < j for i, j in zip(start_vals, end_vals)):
        raise ValueError('The start values need to be smaller then their respective end values!')
    return np.max(start_vals) < np.min(end_vals)
