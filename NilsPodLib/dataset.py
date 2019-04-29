# -*- coding: utf-8 -*-
"""Dataset represents a measurement session of a single sensor.

@author: Nils Roth, Arne Küderle
"""

from pathlib import Path
from typing import Union, Iterable, Optional, Tuple, Dict, TypeVar, Type

import numpy as np
import pandas as pd
from NilsPodLib.datastream import Datastream
from NilsPodLib.header import Header, parse_header
from NilsPodLib.interfaces import CascadingDatasetInterface
from NilsPodLib.utils import path_t, read_binary_file_uint8, convert_little_endian, InvalidInputFileError, \
    RepeatedCalibrationError, inplace_or_copy, datastream_does_not_exist_warning, load_and_check_cal_info
from imucal import CalibrationInfo

T = TypeVar('T')


class Dataset(CascadingDatasetInterface):
    # TODO: Spalte mit Unix timestamp
    # TODO: Potential warning if samplingrate does not fit to rtc
    # TODO: Warning non monotounus counter
    # TODO: Warning if access to not calibrated datastreams
    # TODO: Test calibration
    # TODO: Docu all the things
    # TODO: make baseclass for cascading methods

    def __init__(self, sensor_data: Dict[str, np.ndarray], counter: np.ndarray, info: Header):
        self.counter = counter
        self.info = info
        for k, v in sensor_data.items():
            v = Datastream(v, self.info.sampling_rate_hz, self.info._SENSOR_LEGENDS.get(k, None))
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
    def _datastreams(self) -> Iterable[Datastream]:
        """Iterate through all available datastreams."""
        for i in self.ACTIVE_SENSORS:
            yield i, getattr(self, i)

    def calibrate_imu(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Dataset.

        The calibration can either be provided directly or loaded from a calibration '.json' file.
        """
        calibration = load_and_check_cal_info(calibration)
        s = self.calibrate_acc(calibration, inplace)
        s = s.calibrate_gyro(calibration, inplace=True)
        return s

    def calibrate_acc(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
        # TODO: Allow option to specifiy the unit of the final ds
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.acc, 'acc') is True:
            calibration = load_and_check_cal_info(calibration)
            acc = calibration.calibrate_acc(s.acc.data)
            s.acc.data = acc
            s.acc.is_calibrated = True
        return s

    def calibrate_gyro(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
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

    @property
    def ACTIVE_SENSORS(self) -> Tuple[str]:
        return tuple(self.info.enabled_sensors)

    def downsample(self: T, factor, inplace=False) -> T:
        """Downsample all datastreams by a factor."""
        s = inplace_or_copy(self, inplace)
        for key, val in s._datastreams:
            setattr(s, key, val.downsample(factor))
        return s

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        # TODO: should cut change the start and end date of recording in the header?
        s = inplace_or_copy(self, inplace)

        for key, val in s._datastreams:
            setattr(s, key, val.cut(start, stop, step))
        s.counter = s.counter[start:stop:step]
        return s

    def cut_to_syncregion(self: T, inplace=False) -> T:
        if self.info.is_synchronised is False:
            raise ValueError('Only synchronised Datasets can be cut to the syncregion')
        if self.info.sync_role == 'master':
            return inplace_or_copy(self, inplace)
        return self.cut(self.info.sync_index_start, self.info.sync_index_stop, inplace=inplace)

    def interpolate_dataset(self: T, dataset, inplace=False) -> T:
        raise NotImplementedError('This is currently not working')
        # Todo: fix
        # counterTmp = np.copy(dataset_master_simple.counter)
        # accTmp = np.copy(dataset_master_simple.acc.data)
        # gyroTmp = np.copy(dataset_master_simple.gyro.data)
        # baroTmp = np.copy(dataset_master_simple.baro.data)
        # pressureTmp = np.copy(dataset_master_simple.pressure.data)
        # batteryTmp = np.copy(dataset_master_simple.battery.data)
        #
        # c = 0
        #
        # for i in range(1, len(counterTmp)):
        #     delta = counterTmp[i] - counterTmp[i - 1]
        #     if 1 < delta < 30000:
        #         c = c + 1
        #         counterTmp = self.interpolate_1D(counterTmp, i - 1, delta - 1)
        #         baroTmp = self.interpolate_1D(baroTmp, i - 1, delta - 1)
        #         batteryTmp = self.interpolate_1D(batteryTmp, i - 1, delta - 1)
        #         accTmp = self.interpolate_3d(accTmp, i - 1, delta - 1)
        #         gyroTmp = self.interpolate_3d(gyroTmp, i - 1, delta - 1)
        #         pressureTmp = self.interpolate_3d(pressureTmp, i - 1, delta - 1)
        #
        # if c > 0:
        #     warnings.warn(
        #         "ATTENTION: Dataset was interpolated due to synchronization Error! {} Samples were added!".format(
        #             str(c)))
        #
        # dataset_master_simple.counter = counterTmp
        # dataset_master_simple.gyro.data = gyroTmp
        # dataset_master_simple.pressure.data = pressureTmp
        # dataset_master_simple.baro.data = baroTmp
        # dataset_master_simple.battery.data = batteryTmp
        # return dataset_master_simple

    def data_as_df(self) -> pd.DataFrame:
        dfs = [s.data_as_df() for _, s in self._datastreams]
        return pd.concat(dfs, axis=1)

    def data_as_csv(self, path: path_t) -> None:
        self.data_as_df().to_csv(path, index=False)

    def imu_data_as_df(self) -> pd.DataFrame:
        # TODO: Handle cases were one of the two sensors is not active
        acc_df = self.acc.data_as_df()
        gyro_df = self.gyro.data_as_df()
        return pd.concat([acc_df, gyro_df], axis=1)

    def imu_data_as_csv(self, path: path_t) -> None:
        self.imu_data_as_df().to_csv(path, index=False)


def parse_binary(path: path_t) -> Tuple[Dict[str, np.ndarray],
                                        np.ndarray,
                                        Header]:

    session_header, header_size = parse_header(path)

    sample_size = session_header.sample_size

    data = read_binary_file_uint8(path, sample_size, header_size)
    sensor_data = dict()

    idx = 0
    for sensor in session_header.enabled_sensors:
        bits, channel = session_header._SENSOR_SAMPLE_LENGTH[sensor]
        bits_per_channel = bits // channel
        tmp = np.full((len(data), channel), np.nan)
        for i in range(channel):
            tmp[:, i] = convert_little_endian(np.atleast_2d(data[:, idx:idx + bits_per_channel]).T,
                                              dtype=np.uint32).astype(float)
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

    counter = convert_little_endian(np.atleast_2d(data[:, -4:]).T, dtype=np.uint32).astype(float)

    return sensor_data, counter, session_header

