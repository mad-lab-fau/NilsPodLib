#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""
from typing import Optional

import numpy as np
from scipy import signal


class DataStream:
    data = None
    sampling_rate_hz = None

    def __init__(self, data, sampling_rate: Optional[float] = None):
        self.data = data
        self.sampling_rate_hz = sampling_rate

    def norm(self):
        return np.linalg.norm(self.data, axis=1)

    def normalize(self):
        tmp = np.copy(self.data)
        # TODO: Do this without loop
        for i in range(0, tmp.shape[1]):
            tmp[:, i] = tmp[:, i] / np.max(tmp[:, i])
        return tmp

    def filter_butterworth(self, fc, order, filterType='low'):
        fn = fc / (self.sampling_rate_hz / 2.0)
        b, a = signal.butter(order, fn, btype=filterType)
        return signal.filtfilt(b, a, self.data.T, padlen=150).T
