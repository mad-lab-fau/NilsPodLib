#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import copy
import os

import numpy as np
import pandas as pd
import scipy
from scipy import signal

from NilsPodLib.calibrationData import calibrationData
from NilsPodLib.dataStream import dataStream
from NilsPodLib.parseBinary import parseBinary


class dataset:
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
    calibrationData = None
    size = 0
    isCalibrated = False

    def __init__(self, path):
        self.path = path
        if path.endswith(".bin"):
            [accData, gyrData, baro, pressure, battery, self.counter, self.sync, self.header] = parseBinary(path)
            self.acc = dataStream(accData, self.header.samplingRate_Hz)
            self.gyro = dataStream(gyrData, self.header.samplingRate_Hz)
            self.baro = dataStream(baro, self.header.samplingRate_Hz)
            self.pressure = dataStream(pressure.astype('float'), self.header.samplingRate_Hz)
            self.battery = dataStream(battery, self.header.samplingRate_Hz)
            self.rtc = np.linspace(self.header.unixTime_start, self.header.unixTime_stop, len(self.counter))
            self.size = len(self.counter)

            # todo: add list of calibration files to repository. Ideal Case: For each existing NilPod at least one calibration file exists!
            calibrationFileName = os.path.join(os.path.dirname(__file__), "Calibration/CalibrationFiles/")
            if "84965C0" in self.path:
                calibrationFileName += "NRF52-84965C0.pickle"
                self.calibrationData = calibrationData(calibrationFileName)
            if "92338C81" in self.path:
                calibrationFileName += "NRF52-92338C81.pickle"
                self.calibrationData = calibrationData(calibrationFileName)
        else:
            print("Invalid file tpye")

    def calibrate(self):
        try:
            self.acc.data = (self.calibrationData.Ta * self.calibrationData.Ka * (
                        self.acc.data.T - self.calibrationData.ba)).T
            self.acc.data = np.asarray(self.acc.data)
            self.gyro.data = (self.calibrationData.Tg * self.calibrationData.Kg * (
                        self.gyro.data.T - self.calibrationData.bg)).T
            self.gyro.data = np.asarray(self.gyro.data)
        except:
            # Todo: Use correct static calibration values according to sensor range (this one is hardcoded for 2000dps and 16G)
            self.acc.data = self.acc.data / 2048.0
            self.gyro.data = self.gyro.data / 16.4
            print("No Calibration Data found - Using static Datasheet values for calibration!!!")
        self.isCalibrated = True

    def rotateAxis(self, sensor, x, y, z, sX, sY, sZ):
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
            if 'left' in self.header.sensorPosition:
                print('switching pressure sensors')
                self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
        elif sensor == 'default':
            if 'left' in self.header.sensorPosition:
                self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
                self.acc.data[:, 1] = self.acc.data[:, 1] * -1
                self.gyro.data[:, 0] = self.gyro.data[:, 0] * -1
            # if('right' in self.header.sensorPosition):
            # print "rotating nothing"
            else:
                print("No Position Definition found - Using Name Fallback")
                try:
                    if "92338C81" in self.path:
                        self.pressure.data[:, [0, 1, 2]] = self.pressure.data[:, [2, 1, 0]]
                        self.acc.data[:, 1] = self.acc.data[:, 1] * -1
                        self.gyro.data[:, 0] = self.gyro.data[:, 0] * -1
                except:
                    print("Rotation FAILED")
        else:
            print('unknown sensor, no rotation possible')

    def downSample(self, q):
        dX = scipy.signal.decimate(self.acc.data[:, 0], q)
        dY = scipy.signal.decimate(self.acc.data[:, 1], q)
        dZ = scipy.signal.decimate(self.acc.data[:, 2], q)
        self.acc.data = np.column_stack((dX, dY, dZ))
        dX = scipy.signal.decimate(self.gyro.data[:, 0], q)
        dY = scipy.signal.decimate(self.gyro.data[:, 1], q)
        dZ = scipy.signal.decimate(self.gyro.data[:, 2], q)
        self.gyro.data = np.column_stack((dX, dY, dZ))

    def filterData(self, data, order, fc, fType='lowpass'):
        fn = fc / (self.header.samplingRate_Hz / 2.0)
        b, a = signal.butter(order, fn, btype=fType)
        return signal.filtfilt(b, a, data.T, padlen=150).T

    def cutDataset(self, start, stop):
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

    def exportCSV(self, path):
        if self.isCalibrated:
            accFrame = pd.DataFrame(self.acc.data, columns=['AX [g]', 'AY [g]', 'AZ [g]'])
            gyroFrame = pd.DataFrame(self.gyro.data, columns=['GX [dps]', 'GY [dps]', 'GZ [dps]'])
        else:
            accFrame = pd.DataFrame(self.acc.data, columns=['AX [no unit]', 'AY [no unit]]', 'AZ [no unit]'])
            gyroFrame = pd.DataFrame(self.gyro.data, columns=['GX [no unit]', 'GY [no unit]', 'GZ [no unit]'])

        frame = pd.concat([accFrame, gyroFrame], axis=1)
        frame.to_csv(path, index=False, sep=';')

    @staticmethod
    def interpolate3D(array, idx, num):
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
    def interpolate1D(array, idx, num):
        xx = ((array[idx + 1] - array[idx]) / (num + 1.0))
        for i in range(1, num + 1):
            a = (xx * i) + array[idx]
            array = np.insert(array, idx + i, a)
        return array

    def interpolateDataset(self, dataset):
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
                counterTmp = self.interpolate1D(counterTmp, i - 1, delta - 1)
                baroTmp = self.interpolate1D(baroTmp, i - 1, delta - 1)
                batteryTmp = self.interpolate1D(batteryTmp, i - 1, delta - 1)
                accTmp = self.interpolate3D(accTmp, i - 1, delta - 1)
                gyroTmp = self.interpolate3D(gyroTmp, i - 1, delta - 1)
                pressureTmp = self.interpolate3D(pressureTmp, i - 1, delta - 1)

        if c > 0:
            print(
                "ATTENTION: Dataset was interpolated due to synchronization Error! " + str(c) + " Samples were added!")

        dataset.counter = counterTmp
        dataset.gyro.data = gyroTmp
        dataset.pressure.data = pressureTmp
        dataset.baro.data = baroTmp
        dataset.battery.data = batteryTmp
        return dataset
