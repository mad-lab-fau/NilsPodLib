# -*- coding: utf-8 -*-
"""Dataset represents a measurement session of a single sensor.

@author: Nils Roth, Arne Küderle
"""
import warnings
from distutils.version import StrictVersion
from pathlib import Path
from typing import Union, Iterable, Optional, Tuple, Dict, TypeVar, Type, Sequence, TYPE_CHECKING

import numpy as np
import pandas as pd

from NilsPodLib.consts import SENSOR_SAMPLE_LENGTH
from NilsPodLib.datastream import Datastream
from NilsPodLib.header import Header, parse_header
from NilsPodLib.interfaces import CascadingDatasetInterface
from NilsPodLib.utils import path_t, read_binary_file_uint8, convert_little_endian, InvalidInputFileError, \
    RepeatedCalibrationError, inplace_or_copy, datastream_does_not_exist_warning, load_and_check_cal_info

if TYPE_CHECKING:
    from imucal import CalibrationInfo

T = TypeVar('T')


class Dataset(CascadingDatasetInterface):
    def __init__(self, sensor_data: Dict[str, np.ndarray], counter: np.ndarray, info: Header):
        self.counter = counter
        self.info = info
        for k, v in sensor_data.items():
            v = Datastream(v, self.info.sampling_rate_hz, sensor_type=k)
            setattr(self, k, v)

    @classmethod
    def from_bin_file(cls: Type[T], path: path_t) -> T:
        path = Path(path)
        if not path.suffix == '.bin':
            ValueError('Invalid file type! Only ".bin" files are supported not {}'.format(path))

        sensor_data, counter, info = parse_binary(path)
        s = cls(sensor_data, counter, info)

        s.path = path
        return s

    @classmethod
    def from_csv_file(cls, path: path_t):
        raise NotImplementedError('CSV importer coming soon')

    @property
    def size(self) -> int:
        return len(self.counter)

    @property
    def ACTIVE_SENSORS(self) -> Tuple[str]:
        return tuple(self.info.enabled_sensors)

    @property
    def datastreams(self) -> Iterable[Datastream]:
        """Iterate through all available datastreams."""
        for i in self.ACTIVE_SENSORS:
            yield i, getattr(self, i)

    @property
    def utc_counter(self) -> np.ndarray:
        return self.info.utc_datetime_start_day_midnight.timestamp() + self.counter / self.info.sampling_rate_hz

    @property
    def utc_datetime_counter(self) -> np.ndarray:
        return pd.to_datetime(pd.Series(self.utc_counter * 1000000), utc=True, unit='us').values

    @property
    def time_counter(self) -> np.ndarray:
        return (self.counter - self.counter[0]) / self.info.sampling_rate_hz

    def calibrate_imu(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Dataset.

        The calibration can either be provided directly or loaded from a calibration '.json' file.
        """
        calibration = load_and_check_cal_info(calibration)
        s = self.calibrate_acc(calibration, inplace)
        s = s.calibrate_gyro(calibration, inplace=True)
        return s

    def calibrate_acc(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        # TODO: Allow option to specifiy the unit of the final ds
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.acc, 'acc') is True:
            calibration = load_and_check_cal_info(calibration)
            acc = calibration.calibrate_acc(s.acc.data)
            s.acc.data = acc
            s.acc.is_calibrated = True
        return s

    def calibrate_gyro(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        # TODO: Allow option to specifiy the unit of the final ds
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.gyro, 'gyro') is True:
            calibration = load_and_check_cal_info(calibration)
            gyro = calibration.calibrate_gyro(s.gyro.data)
            s.gyro.data = gyro
            s.gyro.is_calibrated = True
        return s

    def factory_calibrate_imu(self: T, inplace: bool = False) -> T:
        s = self.factory_calibrate_acc(inplace=inplace)
        s = s.factory_calibrate_gyro(inplace=True)

        return s

    def factory_calibrate_gyro(self: T, inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.gyro, 'gyro') is True:
            s.gyro.data /= 2 ** 16 / self.info.gyro_range_dps / 2
            s.gyro.is_calibrated = True
        return s

    def factory_calibrate_acc(self: T, inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.acc, 'acc') is True:
            s.acc.data /= 2 ** 16 / self.info.acc_range_g / 2
            s.acc.is_calibrated = True
        return s

    def factory_calibrate_baro(self: T, inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.baro, 'baro') is True:
            s.baro.data = (s.baro.data + 101325) / 100.0
            s.baro.is_calibrated = True
        return s

    def factory_calibrate_battery(self: T, inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.battery, 'battery') is True:
            s.battery.data = (s.battery.data * 2.0) / 100.0
            s.battery.is_calibrated = True
        return s

    @staticmethod
    def _check_calibration(ds: Datastream, name: str):
        if ds is not None:
            if ds.is_calibrated is True:
                raise RepeatedCalibrationError(name)
            return True
        else:
            datastream_does_not_exist_warning(name, 'calibration')
            return False

    def downsample(self: T, factor, inplace=False) -> T:
        """Downsample all datastreams by a factor."""
        s = inplace_or_copy(self, inplace)
        for key, val in s.datastreams:
            setattr(s, key, val.downsample(factor))
        return s

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        s = inplace_or_copy(self, inplace)

        for key, val in s.datastreams:
            setattr(s, key, val.cut(start, stop, step))
        s.counter = s.counter[start:stop:step]
        return s

    def cut_counter_val(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
                        inplace: bool = False) -> T:
        """Cut the dataset based on values in the counter and not the index."""
        if start:
            start = np.searchsorted(self.counter, start)
        if stop:
            stop = np.searchsorted(self.counter, stop)
        return self.cut(start, stop, step, inplace=inplace)

    def cut_to_syncregion(self: Type[T], end: bool = False, warn_thres: Optional[int] = 30, inplace: bool = False) -> T:
        if self.info.is_synchronised is False:
            raise ValueError('Only synchronised Datasets can be cut to the syncregion')
        if self.info.sync_role == 'master':
            return inplace_or_copy(self, inplace)
        if warn_thres is not None and self._check_sync_packages(warn_thres) is False:
            warnings.warn('The last sync package occured more than {} s before the end of the measurement.'
                          'The last region of the data should not be trusted.'.format(warn_thres))
        end = self.info.sync_index_stop + 1 if end is True else None
        return self.cut(self.info.sync_index_start, end, inplace=inplace)

    def data_as_df(self, datastreams: Optional[Sequence[str]] = None, index: Optional[str] = None) -> pd.DataFrame:
        index_names = {None: 'n_samples', 'counter': 'n_samples', 'time': 't', 'utc': 'utc', 'utc_datetime': 'date'}
        if index and index not in index_names.keys():
            raise ValueError(
                'Supplied value for index ({}) is not allowed. Allowed values: {}'.format(index, index_names.keys()))

        index_name = index_names[index]

        datastreams = datastreams or self.ACTIVE_SENSORS
        dfs = [s.data_as_df() for k, s in self.datastreams if k in datastreams]
        df = pd.concat(dfs, axis=1)

        if index:
            if index != 'counter':
                index += '_counter'
            index = getattr(self, index, None)
            df.index = index
        else:
            df = df.reset_index(drop=True)
        df.index.name = index_name
        return df

    def data_as_csv(self, path: path_t, datastreams: Optional[Iterable[str]] = None,
                    index: Optional[str] = None) -> None:
        self.data_as_df(datastreams=datastreams, index=index).to_csv(path, index=False)

    def imu_data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        return self.data_as_df(datastreams=['acc', 'gyro'], index=index)

    def imu_data_as_csv(self, path: path_t, index: Optional[str] = None) -> None:
        self.imu_data_as_df(index=index).to_csv(path, index=False)

    def _check_sync_packages(self, threshold_s: int = 30) -> bool:
        """Check if the last sync package occurred far from the actual end of the recording.

        This can be the case, if the master stopped sending packages, or if the sensor could not receive any new sync
        info for a prelonged period of time.
        In particular in the latter case, careful review of the data is advised.
        """
        if self.info.sync_role == 'slave':
            if len(self.counter) - self.info.sync_index_stop > threshold_s * self.info.sampling_rate_hz:
                return False
        return True


def parse_binary(path: path_t) -> Tuple[Dict[str, np.ndarray],
                                        np.ndarray,
                                        Header]:
    session_header, header_size = parse_header(path)

    sample_size = session_header.sample_size
    n_samples = session_header.n_samples

    data = read_binary_file_uint8(path, sample_size, header_size, n_samples)
    sensor_data = dict()

    idx = 0
    for sensor in session_header.enabled_sensors:
        bits, channel, dtype = SENSOR_SAMPLE_LENGTH[sensor]
        bits_per_channel = bits // channel
        tmp = np.full((len(data), channel), np.nan)
        for i in range(channel):
            tmp[:, i] = convert_little_endian(np.atleast_2d(data[:, idx:idx + bits_per_channel]).T,
                                              dtype=dtype).astype(float)
            idx += bits_per_channel
        sensor_data[sensor] = tmp

    # Sanity Check:
    if not idx + 4 == data.shape[-1]:
        # TODO: Test if this works as expected
        expected_cols = idx
        all_cols = data.shape[-1] - 4
        raise InvalidInputFileError(
            'The input file has an invalid format. {} data columns expected based on the header, but {} exist.'.format(
                expected_cols, all_cols))

    counter = convert_little_endian(np.atleast_2d(data[:, -4:]).T, dtype=float)

    if session_header.strict_version_firmware >= StrictVersion('0.13.0') and len(counter) != session_header.n_samples:
        warnings.warn('The number of samples in the dataset does not match the number indicated by the header.'
                      'This might indicate a corrupted file')

    return sensor_data, counter, session_header
