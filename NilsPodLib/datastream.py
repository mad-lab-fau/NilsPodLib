# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: Nils Roth, Arne Küderle
"""
import copy
import warnings
from typing import Optional, Iterable, List, TypeVar

import numpy as np
import pandas as pd
from scipy import signal
from scipy.signal import decimate

from NilsPodLib.interfaces import CascadingDatastreamInterface
from NilsPodLib.utils import inplace_or_copy

T = TypeVar('T')

class Datastream(CascadingDatastreamInterface):
    data: np.ndarray
    sampling_rate_hz: float
    columns: List
    is_calibrated: bool = False
    _unit: str

    # TODO: Representatation
    # TODO: Implement inplace vs copy
    # TODO: implement the concept of units

    def __init__(self, data: np.ndarray, sampling_rate: float = 1., columns: Optional[Iterable] = None,
                 unit: Optional[str] = None):
        self.data = data
        self.sampling_rate_hz = float(sampling_rate)
        self._unit = unit
        if columns:
            self.columns = list(columns)
        else:
            self.columns = list(range(data.shape[-1]))

    @property
    def unit(self):
        warnings.warn('Units are not really supported at this point')
        if self.is_calibrated is True:
            return self._unit
        return 'a.u.'

    def __len__(self):
        return len(self.data)

    def norm(self) -> np.ndarray:
        return np.linalg.norm(self.data, axis=1)

    def normalize(self) -> 'Datastream':
        ds = copy.deepcopy(self)
        ds.data /= ds.data.max(axis=0)
        return ds

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)
        sl = slice(start, stop, step)
        s.data = s.data[sl]
        return s

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        """Downsample the datastreams by a factor using a iir filter."""
        s = inplace_or_copy(self, inplace)
        s.data = decimate(s.data, factor, axis=0)
        s.sampling_rate_hz /= factor
        return s

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
