#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import copy
import os
import warnings

import numpy as np
import pandas as pd
import scipy
from scipy import signal

from NilsPodLib.calibration_data import CalibrationData
from NilsPodLib.data_stream import DataStream
from NilsPodLib.parse_binary import parse_binary

ACC = ('acc_' + x for x in 'xyz')
GYR = ('gyr_' + x for x in 'xyz')


class Dataset:
    path = ""
    acc = None
    gyro = None
    baro = None
    pressure = None
    battery = None
    counter = None
    rtc = None
    sync = None
    header = None
    calibration_data = None
    size = 0
    is_calibrated = False

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

        # TODO: add list of calibration files to repository.
        #       Ideal Case: For each existing NilPod at least one calibration file exists!
        # TODO: This should be optional and it should be possible to pass a real file
        calibration_file_name = os.path.join(os.path.dirname(__file__), 'Calibration/CalibrationFiles/')
        if '84965C0' in self.path:
            calibration_file_name += 'NRF52-84965C0.pickle'
            self.calibration_data = CalibrationData(calibration_file_name)
        if '92338C81' in self.path:
            calibration_file_name += 'NRF52-92338C81.pickle'
            self.calibration_data = CalibrationData(calibration_file_name)

    def calibrate(self):
        try:
            self.acc.data = (self.calibration_data.Ta * self.calibration_data.Ka * (
                    self.acc.data.T - self.calibration_data.ba)).T
            self.acc.data = np.asarray(self.acc.data)
            self.gyro.data = (self.calibration_data.Tg * self.calibration_data.Kg * (
                    self.gyro.data.T - self.calibration_data.bg)).T
            self.gyro.data = np.asarray(self.gyro.data)
        except:
            # Todo: Use correct static calibration values according to sensor range
            #       (this one is hardcoded for 2000dps and 16G)
            self.acc.data = self.acc.data / 2048.0
            self.gyro.data = self.gyro.data / 16.4
            warnings.warn('No Calibration Data found - Using static Datasheet values for calibration!')
        self.is_calibrated = True

    def rotate_axis(self, sensor, x, y, z, sX, sY, sZ):
        if sensor == 'gyro':
            tmp = np.copy(self.gyro.data)
            dX = tmp[:, 0]
            dY = tmp[:, 1]
            dZ = tmp[:, 2]
            self.gyro.data[:, x] = dX
            self.gyro.data[:, y] = dY
            self.gyro.data[:, z] = dZ
            self.gyro.data[:, 0] = self.gyro.data[:, 0] * np.sign(sX)
            self.gyro.data[:, 1] = self.gyro.data[:, 1] * np.sign(sY)
            self.gyro.data[:, 2] = self.gyro.data[:, 2] * np.sign(sZ)
        elif sensor == 'acc':
            tmp = np.copy(self.acc.data)
            dX = tmp[:, 0]
            dY = tmp[:, 1]
            dZ = tmp[:, 2]
            self.acc.data[:, x] = dX
            self.acc.data[:, y] = dY
            self.acc.data[:, z] = dZ
            self.acc.data[:, 0] = self.acc.data[:, 0] * np.sign(sX)
            self.acc.data[:, 1] = self.acc.data[:, 1] * np.sign(sY)
            self.acc.data[:, 2] = self.acc.data[:, 2] * np.sign(sZ)
        elif sensor == 'pressure':
            if 'left' in self.header.sensor_position:
                print('switching pressure sensors')
                self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
        elif sensor == 'default':
            if 'left' in self.header.sensor_position:
                self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
                self.acc.data[:, 1] = self.acc.data[:, 1] * -1
                self.gyro.data[:, 0] = self.gyro.data[:, 0] * -1
            else:
                warnings.warn('No Position Definition found - Using Name Fallback')
                try:
                    if '92338C81' in self.path:
                        self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
                        self.acc.data[:, 1] = self.acc.data[:, 1] * -1
                        self.gyro.data[:, 0] = self.gyro.data[:, 0] * -1
                except:
                    # TODO: Replace base exeption and can the try block even fail?? What is going on here anyway.
                    Exception('Rotation FAILED')
        else:
            ValueError('unknown sensor, no rotation possible')

    def down_sample(self, q):
        dX = scipy.signal.decimate(self.acc.data[:, 0], q)
        dY = scipy.signal.decimate(self.acc.data[:, 1], q)
        dZ = scipy.signal.decimate(self.acc.data[:, 2], q)
        self.acc.data = np.column_stack((dX, dY, dZ))
        dX = scipy.signal.decimate(self.gyro.data[:, 0], q)
        dY = scipy.signal.decimate(self.gyro.data[:, 1], q)
        dZ = scipy.signal.decimate(self.gyro.data[:, 2], q)
        self.gyro.data = np.column_stack((dX, dY, dZ))

    def filter_data(self, data, order, fc, fType='lowpass'):
        fn = fc / (self.header.sampling_rate_hz / 2.0)
        b, a = signal.butter(order, fn, btype=fType)
        return signal.filtfilt(b, a, data.T, padlen=150).T

    def cut_dataset(self, start, stop):
        s = copy.copy(self)
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

    def norm(self, data):
        return np.apply_along_axis(np.linalg.norm, 1, data)

    def export_csv(self, path):
        if self.is_calibrated:
            accFrame = pd.DataFrame(self.acc.data, columns=['AX [g]', 'AY [g]', 'AZ [g]'])
            gyroFrame = pd.DataFrame(self.gyro.data, columns=['GX [dps]', 'GY [dps]', 'GZ [dps]'])
        else:
            accFrame = pd.DataFrame(self.acc.data, columns=['AX [no unit]', 'AY [no unit]]', 'AZ [no unit]'])
            gyroFrame = pd.DataFrame(self.gyro.data, columns=['GX [no unit]', 'GY [no unit]', 'GZ [no unit]'])

        frame = pd.concat([accFrame, gyroFrame], axis=1)
        frame.to_csv(path, index=False, sep=';')

    @staticmethod
    def interpolate_3d(array, idx, num):
        # TODO: Das geht doch sicher auch besser oder?
        xx = ((array[idx + 1, 0] - array[idx, 0]) / (num + 1.0))
        yy = ((array[idx + 1, 1] - array[idx, 1]) / (num + 1.0))
        zz = ((array[idx + 1, 2] - array[idx, 2]) / (num + 1.0))
        for i in range(1, num + 1):
            x = (xx * i) + array[idx, 0]
            y = (yy * i) + array[idx, 1]
            z = (zz * i) + array[idx, 2]
            a = [x, y, z]
            array = np.insert(array, idx + i, a, axis=0)
        return array

    @staticmethod
    def interpolate_1D(array, idx, num):
        xx = ((array[idx + 1] - array[idx]) / (num + 1.0))
        for i in range(1, num + 1):
            a = (xx * i) + array[idx]
            array = np.insert(array, idx + i, a)
        return array

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
