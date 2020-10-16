# -*- coding: utf-8 -*-
"""Fundamental Datastream class, which holds any type of sensor_type data and handles basic interactions with it."""

import copy
from typing import Optional, Iterable, List, TypeVar, TYPE_CHECKING

import numpy as np
from scipy.signal import resample

from nilspodlib.consts import SENSOR_LEGENDS, SENSOR_UNITS
from nilspodlib.utils import inplace_or_copy

T = TypeVar("T")

if TYPE_CHECKING:
    import pandas as pd  # noqa: F401


class Datastream:
    """Object representing a single set of data of one sensor_type.

    Usually it is not required to directly interact with the datastream object (besides accessing the data attribute).
    Most important functionality can/should be used via a dataset or session object to manage multiple datasets at once.

    Attributes
    ----------
    data
        The actual data of the sensor_type as `np.array`.
        Can have multiple dimensions depending on the sensor_type.
    sensor
        The name of the sensor_type
    is_calibrated
        If the sensor_type is in a raw format or the expected final output units
    sampling_rate_hz
        The sampling rate of the datastream

    """

    data: np.ndarray
    is_calibrated: bool = False
    sampling_rate_hz: float
    sensor_type: Optional[str]
    _unit: str
    _columns: Optional[List]

    def __init__(
        self,
        data: np.ndarray,
        sampling_rate: float = 1.0,
        columns: Optional[Iterable] = None,
        unit: Optional[str] = None,
        sensor_type: Optional[str] = None,
    ):
        """Get new datastream instance.

        Parameters
        ----------
        data :
            The actual data to be stored in the datastream.
            Should be 2D (even if only 1D data is stored).
            First dimension should be the time axis and second dimension the different data vector entries
            (e.g. acc_x, acc_y, acc_z).
        sampling_rate :
            The sampling rate of the datastream.
            Is used for all calculations that require sampling info.
        columns :
            Optional list of names for the data vector entries.
            Only used to make it easier to understand the content of a datastream.
        unit :
            The unit of the data stored in the datastream.
            This is only used, if `self.is_calibrated` is set to `True`.
        sensor_type :
            Type of sensor_type the data is produced from.
            This allows to automatically get default values for columns and units from :py:mod:`nilspodlib.consts`.

        """
        self.data = data
        self.sampling_rate_hz = float(sampling_rate)
        self.sensor_type = sensor_type
        self._columns = list(columns) if columns else columns
        self._unit = unit

    def __repr__(self):
        """Provide a meaningful str-representation of a Datastream."""
        return "Datastream(sensor_type={}, sampling_rate_hz={}, is_calibrated={}, data={}".format(
            self.sensor_type, self.sampling_rate_hz, self.is_calibrated, self.data
        )

    @property
    def unit(self):
        """Get the unit of the data contained in the datastream.

        This will return either `a.u.` if the datastream is not yet calibrated, the value of `self._unit` if set or
        the default unit from :py:data:`nilspodlib.consts.SENSOR_UNITS`
        """
        if self.is_calibrated is True:
            if self._unit:
                return self._unit
            if self.sensor_type and SENSOR_UNITS.get(self.sensor_type, None):
                return SENSOR_UNITS[self.sensor_type]
        return "a.u."

    @property
    def columns(self):
        """Get the column headers for the data contained in the datastream.

        This will return `self._columns` if set on init or will try to get the default columns from
        :py:data:`nilspodlib.consts.SENSOR_LEGENDS`.
        If none of the above is applicable the columns wil be numbered starting with 0.
        """
        if self._columns:
            return self._columns
        if self.sensor_type:
            if SENSOR_LEGENDS.get(self.sensor_type, None):
                return list(SENSOR_LEGENDS[self.sensor_type])
        return list(range(self.data.shape[-1]))

    def __len__(self):
        """Length of the datastream in samples."""
        return len(self.data)

    def norm(self) -> np.ndarray:
        """Calculate the norm of the data along the last axis.

        This will provide the vector norm for each time point.
        """
        return np.linalg.norm(self.data, axis=-1)

    def normalize(self) -> "Datastream":
        """Get the normalized data.

        Normalization is performed by dividing the data by its maximum absolute value.
        This will always return a copy of the datastream.
        """
        ds = copy.deepcopy(self)
        ds.data /= np.abs(ds.data).max(axis=0)
        return ds

    def cut(
        self: T,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        inplace: bool = False,
    ) -> T:
        """Cut the datastream.

        This is equivalent to applying the following slicing to the data of the datastream: `array[start:stop:step]`

        Parameters
        ----------
        start :
            Start index
        stop :
            Stop index
        step :
            Step size of the cut
        inplace :
            If True this methods modifies the current datastream object. If False, a copy of the datastream is returned.

        """
        s = inplace_or_copy(self, inplace)
        sl = slice(start, stop, step)
        s.data = s.data[sl]
        return s

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        """Downsample the datastreams by a factor.

        Parameters
        ----------
        factor :
            Factor by which the datastream should be downsampled.
        inplace :
            If True this methods modifies the current datastream object. If False, a copy of the datastream is returned.

        """
        s = inplace_or_copy(self, inplace)
        s.data = resample(s.data, len(s.data) // factor, axis=0)
        s.sampling_rate_hz /= factor
        return s

    def data_as_df(self, index_as_time: bool = True, include_units=False) -> "pd.DataFrame":
        """Export the datastream as pandas DataFrame using `self.columns` as colummns.

        Parameters
        ----------
        index_as_time :
            If True the index will be divided by the sampling rate to represent time since start of the measurement
        include_units :
            If True the column names will have the unit of the datastream concatenated with an (Default value = False)

        Returns
        -------
        DataFrame

        """
        import pandas as pd  # noqa: F811

        columns = self.columns
        if include_units is True:
            columns = ["{}_{}".format(c, self.unit) for c in columns]
        df = pd.DataFrame(self.data, columns=columns)
        if index_as_time:
            df.index /= self.sampling_rate_hz
            df.index.name = "t"
        return df
