#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import struct
import warnings
from pathlib import Path
from typing import Union, Iterable, Optional, Tuple, Dict

import numpy as np
import pandas as pd
from NilsPodLib.datastream import Datastream
from NilsPodLib.header import Header
from NilsPodLib.utils import path_t, read_binary_file_uint8, convert_little_endian, InvalidInputFileError, \
    RepeatedCalibrationError, inplace_or_copy, datastream_does_not_exist_warning
from imucal import CalibrationInfo


class Dataset:
    path: path_t
    acc: Optional[Datastream]
    gyro: Optional[Datastream]
    mag: Optional[Datastream]
    baro: Optional[Datastream]
    analog: Optional[Datastream]
    ecg: Optional[Datastream]
    ppg: Optional[Datastream]
    battery: Optional[Datastream]
    counter: np.ndarray
    rtc: np.ndarray
    info: Header

    imu_is_calibrated: bool = False
    baro_is_calibrated: bool = False
    battery_is_calibrated: bool = False

    ACTIVE_SENSORS: tuple

    # TODO: Add alternative consturctorsg
    # TODO: Spalte mit Unix timestamp
    # TODO: Potential warning if samplingrate does not fit to rtc
    # TODO: Warning non monotounus counter
    # TODO: Warning if access to not calibrated datastreams
    # TODO: Test calibration

    def __init__(self, path: Union[Path, str]):
        path = Path(path)
        if not path.suffix == '.bin':
            ValueError('Invalid file type! Only ".bin" files are supported not {}'.format(path))

        self.path = path
        sensor_data, self.counter, self.info = parse_binary(self.path)
        active_sensors = []
        for k, v in sensor_data.items():
            if v is not None:
                active_sensors.append(k)
                v = Datastream(v, self.info.sampling_rate_hz, self.info._SENSOR_LEGENDS.get(k, None))
            setattr(self, k, v)
        self.ACTIVE_SENSORS = tuple(active_sensors)

    @property
    def size(self) -> int:
        return len(self.counter)

    @property
    def _DATASTREAMS(self) -> Iterable[Datastream]:
        """Iterate through all available datastreams."""
        for i in self.ACTIVE_SENSORS:
            yield i, getattr(self, i)

    def calibrate_imu(self, calibration: Optional[Union[CalibrationInfo, path_t]] = None,
                      inplace: bool = False, supress_warning=False) -> 'Dataset':
        """Apply a calibration to the Dataset.

        The calibration can either be provided directly or loaded from a calibration '.json' file.
        If no calibration info is provided, factory calibration is applied.
        """
        s = inplace_or_copy(self, inplace)

        if calibration is None:
            s.factory_calibrate_imu()
            if supress_warning is not True:
                warnings.warn('No Calibration Data provided - Using static Datasheet values for calibration!')
            return s
        elif isinstance(calibration, (Path, str)):
            calibration = CalibrationInfo.from_json_file(calibration)

        # TODO: Handle cases were either Acc or gyro are disabled
        acc, gyro = calibration.calibrate(s.acc.data, s.gyro.data)
        s.acc.data = acc
        s.gyro.data = gyro
        s.imu_is_calibrated = True

        return s

    def factory_calibrate_imu(self):
        """Perform a factory calibration based values extracted from the sensors datasheet.

        Note: It is highly recommended to perform a real calibration to use the sensordata in any meaningful context
        """
        # Todo: Use correct static calibration values according to sensor range
        #       (this one is hardcoded for 2000dps and 16G)
        self.acc.data /= 2048.0
        self.gyro.data /= 16.4
        self.imu_is_calibrated = True

    def factory_calibrate_baro(self, inplace: bool = False) -> 'Dataset':
        s = inplace_or_copy(self, inplace)

        if s.baro.is_calibrated is True:
            raise RepeatedCalibrationError('baro')

        return s._factory_calibrate_baro(s)

    def factory_calibrate_battery(self, inplace: bool = False) -> 'Dataset':
        s = inplace_or_copy(self, inplace)

        if s.battery.is_calibrated is True:
            raise RepeatedCalibrationError('battery')

        return s._factory_calibrate_battery(s)

    @staticmethod
    def _factory_calibrate_baro(dataset: 'Dataset') -> 'Dataset':
        if dataset.baro is not None:
            dataset.baro = (dataset.baro + 101325) / 100.0
            dataset.baro.is_calibrated = True
        else:
            datastream_does_not_exist_warning('baro', 'calibration')
        return dataset

    @staticmethod
    def _factory_calibrate_battery(dataset: 'Dataset'):
        if dataset.battery is not None:
            dataset.battery = (dataset.battery * 2.0) / 100.0
            dataset.battery.is_calibrated = True
        else:
            datastream_does_not_exist_warning('battery', 'calibration')
        return dataset

    def downsample(self, factor, inplace=False) -> 'Dataset':
        """Downsample all datastreams by a factor."""
        s = inplace_or_copy(self, inplace)
        for key, val in s._DATASTREAMS:
            setattr(s, key, val.downsample(factor))
        return s

    def cut(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> 'Dataset':
        s = inplace_or_copy(self, inplace)

        for key, val in s._DATASTREAMS:
            setattr(s, key, val.cut(start, stop, step))
        s.sync = s.sync[start:stop: step]
        s.counter = s.counter[start:stop:step]
        s.rtc = s.rtc[start:stop:step]
        return s

    def interpolate_dataset(self, dataset, inplace=False):
        # Todo: fix
        counterTmp = np.copy(dataset.counter)
        accTmp = np.copy(dataset.acc.data)
        gyroTmp = np.copy(dataset.gyro.data)
        baroTmp = np.copy(dataset.baro.data)
        pressureTmp = np.copy(dataset.pressure.data)
        batteryTmp = np.copy(dataset.battery.data)

        c = 0

        for i in range(1, len(counterTmp)):
            delta = counterTmp[i] - counterTmp[i - 1]
            if 1 < delta < 30000:
                c = c + 1
                counterTmp = self.interpolate_1D(counterTmp, i - 1, delta - 1)
                baroTmp = self.interpolate_1D(baroTmp, i - 1, delta - 1)
                batteryTmp = self.interpolate_1D(batteryTmp, i - 1, delta - 1)
                accTmp = self.interpolate_3d(accTmp, i - 1, delta - 1)
                gyroTmp = self.interpolate_3d(gyroTmp, i - 1, delta - 1)
                pressureTmp = self.interpolate_3d(pressureTmp, i - 1, delta - 1)

        if c > 0:
            warnings.warn(
                "ATTENTION: Dataset was interpolated due to synchronization Error! {} Samples were added!".format(
                    str(c)))

        dataset.counter = counterTmp
        dataset.gyro.data = gyroTmp
        dataset.pressure.data = pressureTmp
        dataset.baro.data = baroTmp
        dataset.battery.data = batteryTmp
        return dataset

    def imu_data_as_df(self) -> pd.DataFrame:
        acc_df = self.acc.data_as_df()
        gyro_df = self.gyro.data_as_df()
        return pd.concat([acc_df, gyro_df], axis=1)

    def imu_data_as_csv(self, path):
        self.imu_data_as_df().to_csv(path, index=False)


def parse_binary(path: path_t) -> Tuple[Dict[str, np.ndarray],
                                        np.ndarray,
                                        Header]:
    with open(path, 'rb') as f:
        # TODO: Don't read the full file here, but only the header to improve performance
        data = f.read()

    header_size = data[0]

    data = bytearray(data)
    header_bytes = np.asarray(struct.unpack(str(header_size) + 'b', data[0:header_size]), dtype=np.uint8)
    session_header = Header(header_bytes[1:header_size])

    sample_size = session_header.sample_size

    data = read_binary_file_uint8(path, sample_size, header_size)
    sensor_data = dict()

    idx = 0
    for sensor in session_header._SENSOR_FLAGS:
        if getattr(session_header, sensor + '_enabled') is True:
            bits, channel = session_header._SENSOR_SAMPLE_LENGTH[sensor]
            bits_per_channel = bits // channel
            tmp = np.full((len(data), channel), np.nan)
            for i in range(channel):
                tmp[:, i] = convert_little_endian(np.atleast_2d(data[:, idx:idx + bits_per_channel]).T,
                                                  dtype=np.uint32).astype(float)
                idx += bits_per_channel
        else:
            tmp = None
        sensor_data[sensor] = tmp

    # Sanity Check:
    if idx + 4 != data.shape[-1]:
        # TODO: Test if this works as expected
        expected_cols = idx
        all_cols = data.shape[-1] - 4
        raise InvalidInputFileError(
            'The input file has an invalid format. {} data columns expected based on the header, but {} exist.'.format(
                expected_cols, all_cols))

    counter = convert_little_endian(np.atleast_2d(data[:, -4:]).T, dtype=np.uint32).astype(float)

    return sensor_data, counter, session_header
