#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import numpy as np
from scipy import signal


class dataStream:
    data = None
    samplingRate_Hz = None

    def __init__(self, data, sR=None):
        self.data = data
        self.samplingRate_Hz = sR

    def norm(self):
        return np.linalg.norm(self.data, axis=1)

    def normalize(self):
        tmp = np.copy(self.data)
        for i in range(0, tmp.shape[1]):
            tmp[:, i] = tmp[:, i] / np.max(tmp[:, i])
        return tmp

    def filter_butterworth(self, fc, order, filterType='low'):
        fn = fc / (self.samplingRate_Hz / 2.0)
        b, a = signal.butter(order, fn, btype=filterType)
        return signal.filtfilt(b, a, self.data.T, padlen=150).T
