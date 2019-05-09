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
    from imucal import CalibrationInfo


path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')


def convert_little_endian(byte_list, dtype=int):
    byte_list = np.array(byte_list).astype(np.uint32)
    number = byte_list[0]
    for i, v in enumerate(byte_list[1:]):
        number |= v << int(8 * (i + 1))
    return number.astype(dtype)


def read_binary_uint8(data_bytes, packet_size, expected_samples):
    if expected_samples * packet_size > len(data_bytes):
        expected_samples = len(data_bytes) // packet_size
    data_bytes = data_bytes[:expected_samples * packet_size]
    data = np.reshape(data_bytes, (expected_samples, int(packet_size)))
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


class VersionError(Exception):
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
    from imucal import CalibrationInfo
    if isinstance(calibration, (Path, str)):
        calibration = CalibrationInfo.from_json_file(calibration)
    if not isinstance(calibration, CalibrationInfo):
        raise ValueError('No valid CalibrationInfo object provided')
    return calibration


def validate_existing_overlap(start_vals: np.ndarray, end_vals: np.ndarray) -> bool:
    if not all(i < j for i, j in zip(start_vals, end_vals)):
        raise ValueError('The start values need to be smaller then their respective end values!')
    return np.max(start_vals) < np.min(end_vals)


def legacy_support_check(version: StrictVersion, as_warning: bool = False):
    msg = None
    if version < StrictVersion('0.11.2'):
        msg = 'You are using a version ({}) previous to 0.11.2. This version is not supported!'.format(version)
    elif StrictVersion('0.11.2') <= version < StrictVersion('0.12.0'):
        msg = 'You are using a version ({}) which is only supported by legacy support.' \
              'Use `NilsPodLib.legacy.convert_11_2` to update the binary format to a newer version.'.format(version)

    if msg:
        if as_warning is True:
            warnings.warn(msg)
        else:
            raise VersionError(msg)
