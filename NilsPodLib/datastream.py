# -*- coding: utf-8 -*-
"""Fundamental Datastream class, which holds any type of sensor data and handles basic interactions with it."""

import copy
from typing import Optional, Iterable, List, TypeVar, TYPE_CHECKING

import numpy as np
from scipy.signal import decimate

from NilsPodLib.consts import SENSOR_LEGENDS, SENSOR_UNITS
from NilsPodLib.utils import inplace_or_copy

T = TypeVar('T')

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401


class Datastream:
    """Object representing a single set of data of one sensor.

    Usually it is not required to directly interact with the datastream object (besides accessing the data attribute).
    Most important functionality can/should be used via a dataset or session object to manage multiple datasets at once.
    """

    data: np.ndarray
    is_calibrated: bool = False
    sampling_rate_hz: float
    sensor: Optional[str]
    _unit: str
    _columns: Optional[List]

    def __init__(self, data: np.ndarray, sampling_rate: float = 1., columns: Optional[Iterable] = None,
                 unit: Optional[str] = None, sensor_type: Optional[str] = None):
        """Get new datastream instance.

        Args:
            data: The actual data to be stored in the datastream. Should be 2D (even if only 1D data is stored).
                First dimension should be the time axis and second dimension the different data vector entries
                (e.g. acc_x, acc_y, acc_z)
            sampling_rate: The sampling rate of the datastream. Is used for all calculations that require sampling info
            columns: Optional list of names for the data vector entries. Only used to make it easier to understand the
                content of a datastream
            unit: The unit of the data stored in the datastream.
                This is only used, if `self.is_calibrated` is set to `True`
            sensor_type: Type of sensor the data is produced from. This allows to automatically get default values for
                columns and units from :py:mod:`NilsPodLib.consts
        """
        self.data = data
        self.sampling_rate_hz = float(sampling_rate)
        self.sensor = sensor_type
        self._columns = list(columns) if columns else columns
        self._unit = unit

    def __repr__(self):
        return 'Datastream(sensor={}, sampling_rate_hz={}, is_calibrated={}, data={}'.format(self.sensor,
                                                                                             self.sampling_rate_hz,
                                                                                             self.is_calibrated,
                                                                                             self.data)

    @property
    def unit(self):
        """Get the unit of the data contained in the datastream.

        This will return either `a.u.` if the datastream is not yet calibrated, the value of `self._unit` if set or
        the default unit from :py:data:`NilsPodLib.consts.SENSOR_UNITS`
        """
        if self.is_calibrated is True:
            if self._unit:
                return self._unit
            if self.sensor and SENSOR_UNITS.get(self.sensor, None):
                return SENSOR_UNITS[self.sensor]
        return 'a.u.'

    @property
    def columns(self):
        """Get the column headers for the data contained in the datastream.

        This will return `self._columns` if set on init or will try to get the default columns from
        :py:data:`NilsPodLib.consts.SENSOR_LEGENDS`.
        If none of the above is applicable the columns wil be numbered starting with 0.
        """
        if self._columns:
            return self._columns
        elif self.sensor:
            if SENSOR_LEGENDS.get(self.sensor, None):
                return list(SENSOR_LEGENDS[self.sensor])
        return list(range(self.data.shape[-1]))

    def __len__(self):
        return len(self.data)

    def norm(self) -> np.ndarray:
        """Calculate the norm of the data along the last axis.

        This will provide the vector norm for each time point.
        """
        return np.linalg.norm(self.data, axis=-1)

    def normalize(self) -> 'Datastream':
        """Get the normalized data.

        Normalization is performed by dividing the data by its maximum absolute value.
        """
        ds = copy.deepcopy(self)
        ds.data /= np.abs(ds.data).max(axis=0)
        return ds

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        """Cut the datastream.

        This is equivalent to applying the following slicing to the data of the datastream: `array[start:stop:step]`

        Args:
            start: Start index
            stop: Stop index
            step: Step size of the cut
            inplace: If True this methods modifies the current datastream object. If False, a copy of the datastream

        """
        s = inplace_or_copy(self, inplace)
        sl = slice(start, stop, step)
        s.data = s.data[sl]
        return s

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        """Downsample all datastreams by a factor.

        This applies `scipy.signal.decimate` to all datastreams and the counter of the dataset.

        Args:
            factor: Factor by which the dataset should be downsampled.
            inplace: If True this methods modifies the current datastream object. If False, a copy of the datastream

        """
        s = inplace_or_copy(self, inplace)
        s.data = decimate(s.data, factor, axis=0)
        s.sampling_rate_hz /= factor
        return s

    def data_as_df(self, index_as_time: bool = True) -> 'pd.DataFrame':
        """Return the datastream as pandas Dataframe.

        This will use `self.columns` as columns for the dataframe.

        Args:
            index_as_time: If True the index will be divided by the sampling rate to represent time since start of the
                measurement.
        """
        import pandas as pd  # noqa: F811
        df = pd.DataFrame(self.data, columns=self.columns)
        if index_as_time:
            df.index /= self.sampling_rate_hz
            df.index.name = 't'
        return df
