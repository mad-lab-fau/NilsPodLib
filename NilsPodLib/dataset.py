#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import copy
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import signal

from NilsPodLib.data_stream import DataStream
from NilsPodLib.parse_binary import parse_binary

ACC = tuple('acc_' + x for x in 'xyz')
GYR = tuple('gyr_' + x for x in 'xyz')


class Dataset:
    path = ""
    acc: DataStream
    gyro: DataStream
    baro: DataStream
    pressure: DataStream
    battery: DataStream
    counter: np.ndarray
    rtc: np.ndarray
    sampling_rate_hz: float
    sync = None
    header = None
    calibration_data = None
    size: int
    is_calibrated: bool = False

    def __init__(self, path):
        if not path.endswith('.bin'):
            ValueError('Invalid file type! Only ".bin" files are supported not {}'.format(path))

        self.path = path
        accData, gyrData, baro, pressure, battery, self.counter, self.sync, self.header = parse_binary(self.path)
        self.acc = DataStream(accData, self.header.sampling_rate_hz, legend=ACC)
        self.gyro = DataStream(gyrData, self.header.sampling_rate_hz, legend=GYR)
        self.baro = DataStream(baro, self.header.sampling_rate_hz)
        self.pressure = DataStream(pressure.astype('float'), self.header.sampling_rate_hz)
        self.battery = DataStream(battery, self.header.sampling_rate_hz)
        self.rtc = np.linspace(self.header.unix_time_start, self.header.unix_time_stop, len(self.counter))
        self.size = len(self.counter)
        self.sampling_rate_hz = self.header.sampling_rate_hz

    def calibrate(self):
        try:
            # TODO: Make use of new calibration lib
            self.acc.data = (self.calibration_data.Ta * self.calibration_data.Ka * (
                    self.acc.data.T - self.calibration_data.ba)).T
            self.acc.data = np.asarray(self.acc.data)
            self.gyro.data = (self.calibration_data.Tg * self.calibration_data.Kg * (
                    self.gyro.data.T - self.calibration_data.bg)).T
            self.gyro.data = np.asarray(self.gyro.data)
            self.is_calibrated = True
        except:
            self.factory_calibration()
            warnings.warn('No Calibration Data found - Using static Datasheet values for calibration!')

    def factory_calibration(self):
        """Perform a factory calibration based values extracted from the sensors datasheet.

        Note: It is highly recommended to perform a real calibration to use the sensordata in any meaningful context
        """
        # Todo: Use correct static calibration values according to sensor range
        #       (this one is hardcoded for 2000dps and 16G)
        self.acc.data = self.acc.data / 2048.0
        self.gyro.data = self.gyro.data / 16.4
        self.is_calibrated = True

    def down_sample(self, q):
        dX = scipy.signal.decimate(self.acc.data[:, 0], q)
        dY = scipy.signal.decimate(self.acc.data[:, 1], q)
        dZ = scipy.signal.decimate(self.acc.data[:, 2], q)
        self.acc.data = np.column_stack((dX, dY, dZ))
        dX = scipy.signal.decimate(self.gyro.data[:, 0], q)
        dY = scipy.signal.decimate(self.gyro.data[:, 1], q)
        dZ = scipy.signal.decimate(self.gyro.data[:, 2], q)
        self.gyro.data = np.column_stack((dX, dY, dZ))

    def cut_dataset(self, start, stop) -> 'Dataset':
        s = copy.deepcopy(self)
        s.sync = s.sync[start:stop]
        s.counter = s.counter[start:stop]
        s.acc.data = s.acc.data[start:stop]
        s.gyro.data = s.gyro.data[start:stop]
        s.baro.data = s.baro.data[start:stop]
        s.pressure.data = s.pressure.data[start:stop]
        s.battery.data = s.battery.data[start:stop]
        s.rtc = s.rtc[start:stop]
        s.size = len(s.counter)
        return s

    def interpolate_dataset(self, dataset):
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

