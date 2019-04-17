#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""
import copy
from typing import Optional, Iterable, List

import numpy as np
import pandas as pd
from scipy import signal


class Datastream:
    data: np.ndarray
    sampling_rate_hz: float
    columns: List

    def __init__(self, data: np.ndarray, sampling_rate: Optional[float] = 1., columns: Optional[Iterable] = None):
        self.data = data
        self.sampling_rate_hz = float(sampling_rate)
        if columns:
            self.columns = list(columns)
        else:
            self.columns = list(range(data.shape[-1]))

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.data, axis=1)

    def normalize(self):
        return self.data / self.data.max(axis=0)

    def cut(self, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None):
        ds = copy.deepcopy(self)
        sl = slice(start, stop, step)
        ds.data = ds.data[sl]
        return ds

    def filter_butterworth(self, fc, order, filterType='low'):
        fn = fc / (self.sampling_rate_hz / 2.0)
        b, a = signal.butter(order, fn, btype=filterType)
        return signal.filtfilt(b, a, self.data.T, padlen=150).T

    def data_as_df(self, index_as_time: bool = True) -> pd.DataFrame:
        df = pd.DataFrame(self.data, columns=self.columns)
        if index_as_time:
            df.index /= self.sampling_rate_hz
            df.index.name = 't'
        return df
