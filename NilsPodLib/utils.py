#!/usr/bin/python3
# -*- coding: utf-8 -*-
import copy
import warnings
from typing import TypeVar, Union

import numpy as np
from pathlib import Path

from imucal import CalibrationInfo

path_t = TypeVar('path_t', str, Path)
T = TypeVar('T')


def convert_little_endian(byte_list, dtype=int):
    byte_list = np.array(byte_list).astype(dtype)
    number = byte_list[0]
    for i, v in enumerate(byte_list[1:]):
        number |= v << int(8 * (i + 1))
    return number


def read_binary_file_uint8(path, packet_size, skip_header_bytes):
    with open(path, 'rb') as f:
        f.seek(skip_header_bytes)  # skip Header bytes
        data = np.fromfile(f, dtype=np.dtype('B'))
    data = data[0:(int(len(data) / packet_size) * packet_size)]
    data = np.reshape(data, (int(len(data) / packet_size), int(packet_size)))
    return data


def read_binary_file_int16(path, packet_size, skip_header_bytes):
    with open(path, 'rb') as f:
        f.seek(skip_header_bytes)  # skip Header bytes
        data = np.fromfile(f, dtype=np.dtype('i2'))
    data = data[0:(int(len(data) / int(packet_size / 2)) * int(packet_size / 2))]
    data = np.reshape(data, (int(len(data) / (packet_size / 2)), int(packet_size / 2)))
    return data


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


def load_and_check_cal_info(calibration: Union[CalibrationInfo, path_t]) -> CalibrationInfo:
    if isinstance(calibration, (Path, str)):
        calibration = CalibrationInfo.from_json_file(calibration)
    if not isinstance(calibration, CalibrationInfo):
        raise ValueError('No valid CalibrationInfo object provided')
    return calibration


def validate_existing_overlap(start_vals: np.ndarray, end_vals: np.ndarray) -> bool:
    if not all(i < j for i, j in zip(start_vals, end_vals)):
        raise ValueError('The start values need to be smaller then their respective end values!')
    return np.max(start_vals) < np.min(end_vals)


