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
from scipy.signal import decimate

from NilsPodLib.calibration_utils import find_closest_calibration_to_date
from NilsPodLib.consts import SENSOR_SAMPLE_LENGTH
from NilsPodLib.datastream import Datastream
from NilsPodLib.header import Header
from NilsPodLib.interfaces import CascadingDatasetInterface
from NilsPodLib.utils import path_t, read_binary_uint8, convert_little_endian, InvalidInputFileError, \
    RepeatedCalibrationError, inplace_or_copy, datastream_does_not_exist_warning, load_and_check_cal_info, \
    get_header_and_data_bytes, get_strict_version_from_header_bytes
from NilsPodLib.legacy import legacy_support_check

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401

T = TypeVar('T')


class Dataset(CascadingDatasetInterface):
    def __init__(self, sensor_data: Dict[str, np.ndarray], counter: np.ndarray, info: Header):
        self.counter = counter
        self.info = info
        for k, v in sensor_data.items():
            v = Datastream(v, self.info.sampling_rate_hz, sensor_type=k)
            setattr(self, k, v)

    @classmethod
    def from_bin_file(cls: Type[T], path: path_t, legacy_error: bool = True) -> T:
        """Create a new Dataset from a valid .bin file.

        Args:
            path: Path to the file
            legacy_error: The method checks, if the binary file was created with a compatible Firmware Version.
                If `legacy_error` is True, this will raise an error, if an unsupported version is detected.
                If `legacy_error` is False, only a warning will be displayed.

        Raises:
             VersionError: If unsupported FirmwareVersion is detected and `legacy_error` is True

        """
        path = Path(path)
        if not path.suffix == '.bin':
            ValueError('Invalid file type! Only ".bin" files are supported not {}'.format(path))

        sensor_data, counter, info = parse_binary(path, legacy_error=legacy_error)
        s = cls(sensor_data, counter, info)

        s.path = path
        return s

    @classmethod
    def from_csv_file(cls, path: path_t):
        """Create a new Dataset from a valid .csv file.

        Args:
            path: Path to the file

        Notes:
            This is planned but not yet supported

        """
        raise NotImplementedError('CSV importer coming soon')

    @property
    def size(self) -> int:
        """Get the number of samples in the Dataset."""
        return len(self.counter)

    @property
    def active_sensors(self) -> Tuple[str]:
        """Get the enabled sensors in the dataset."""
        return tuple(self.info.enabled_sensors)

    @property
    def datastreams(self) -> Iterable[Datastream]:
        """Iterate through all available datastreams."""
        for i in self.active_sensors:
            yield i, getattr(self, i)

    @property
    def utc_counter(self) -> np.ndarray:
        """Counter as utc timestamps."""
        return self.info.utc_datetime_start_day_midnight.timestamp() + self.counter / self.info.sampling_rate_hz

    @property
    def utc_datetime_counter(self) -> np.ndarray:
        """Counter as np.datetime64 in UTC timezone."""
        return pd.to_datetime(pd.Series(self.utc_counter * 1000000), utc=True, unit='us').values

    @property
    def time_counter(self) -> np.ndarray:
        """Counter in seconds since first sample."""
        return (self.counter - self.counter[0]) / self.info.sampling_rate_hz

    def calibrate_imu(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Acc and Gyro datastreams.

        The final units of the datastreams will depend on the used calibration values, but must likely they will be "g"
        for the Acc and "dps" (degrees per second) for the Gyro.

        Args:
            calibration: calibration object or path to .json file, that can be used to create one.
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                datastream objects is created

        Notes:
            This just combines `calibrate_acc` and `calibrate_gyro`.

        """
        s = inplace_or_copy(self, inplace)
        check = [self._check_calibration(s.acc, 'acc'), self._check_calibration(s.gyro, 'gyro')]
        if all(check):
            calibration = load_and_check_cal_info(calibration)
            acc, gyro = calibration.calibrate(s.acc.data, s.gyro.data)
            s.acc.data = acc
            s.gyro.data = gyro
            s.acc.is_calibrated = True
            s.acc._unit = calibration.ACC_UNIT
            s.gyro.is_calibrated = True
            s.gyro._unit = calibration.GYRO_UNIT
        return s

    def calibrate_acc(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Acc datastream.

        The final units of the datastream will depend on the used calibration values, but must likely they will be "g"
        for Acc.

        Args:
            calibration: calibration object or path to .json file, that can be used to create one.
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.acc, 'acc') is True:
            calibration = load_and_check_cal_info(calibration)
            acc = calibration.calibrate_acc(s.acc.data)
            s.acc.data = acc
            s.acc.is_calibrated = True
            s.acc._unit = calibration.ACC_UNIT
        return s

    def calibrate_gyro(self: T, calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        """Apply a calibration to the Gyro datastream.

        The final units of the datastreams will depend on the used calibration values, but must likely they will be
        "dps" (degrees per second) for the Gyro.

        Args:
            calibration: calibration object or path to .json file, that can be used to create one.
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.gyro, 'gyro') is True:
            calibration = load_and_check_cal_info(calibration)
            gyro = calibration.calibrate_gyro(s.gyro.data)
            s.gyro.data = gyro
            s.gyro.is_calibrated = True
            s.gyro._unit = calibration.GYRO_UNIT
        return s

    def factory_calibrate_imu(self: T, inplace: bool = False) -> T:
        """Apply a calibration to the Acc and Gyro datastreams.

        The values used for that are taken from the datasheet of the sensor and are likely not to be accurate.
        For any tasks requiring precise sensor outputs, `calibrate_imu` should be used with measured calibration
        values.

        The final units of the output will be "g" for the Acc and "dps" (degrees per second) for the Gyro.

        Args:
             inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        Notes:
            This just combines `factory_calibrate_acc` and `factory_calibrate_gyro`.

        """
        s = self.factory_calibrate_acc(inplace=inplace)
        s = s.factory_calibrate_gyro(inplace=True)

        return s

    def factory_calibrate_gyro(self: T, inplace: bool = False) -> T:
        """Apply a factory calibration to the Gyro datastream.

        The values used for that are taken from the datasheet of the sensor and are likely not to be accurate.
        For any tasks requiring precise sensor outputs, `calibrate_gyro` should be used with measured calibration
        values.

        The final units of the output will be "dps" (degrees per second) for the Gyro.

        Args:
             inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.gyro, 'gyro') is True:
            s.gyro.data /= 2 ** 16 / self.info.gyro_range_dps / 2
            s.gyro.is_calibrated = True
        return s

    def factory_calibrate_acc(self: T, inplace: bool = False) -> T:
        """Apply a factory calibration to the Acc datastream.

        The values used for that are taken from the datasheet of the sensor and are likely not to be accurate.
        For any tasks requiring precise sensor outputs, `calibrate_acc` should be used with measured calibration
        values.

        The final units of the output will be "g" for the Acc.

        Args:
             inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.acc, 'acc') is True:
            s.acc.data /= 2 ** 16 / self.info.acc_range_g / 2
            s.acc.is_calibrated = True
        return s

    def factory_calibrate_baro(self: T, inplace: bool = False) -> T:
        """Apply a calibration to the Baro datastream.

        The values used for that are taken from the datasheet of the sensor and are likely not to be accurate.
        In general, if baro measurements are used to estimate elevation, the estimation should be calibrated relative to
        a reference altitude.

        The final units of the output will be "millibar" (equivalent to Hectopacal) for the Baro.

        Args:
             inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.baro, 'baro') is True:
            s.baro.data = (s.baro.data + 101325) / 100.0
            s.baro.is_calibrated = True
        return s

    def factory_calibrate_temperature(self: Type[T], inplace: bool = False):
        """Apply a factory calibration to the temperature datastream.

        The values used for that are taken from the datasheet of the sensor

        The final unit is Celsius.

        Args:
             inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created
        """
        s = inplace_or_copy(self, inplace)
        if self._check_calibration(s.temperature, 'temperature') is True:
            s.temperature.data = s.temperature.data * (2 ** -9) + 23
            s.temperature.is_calibrated = True
        return s

    @staticmethod
    def _check_calibration(ds: Optional[Datastream], name: str):
        """Check if a specific datastream is already marked as calibrated, or if the datastream does not exist.

        In case the datastream is already calibrated a `RepeatedCalibrationError` is raised.
        In case the datastream does not exist, a warning is raised.

        Args:
            ds: datastream object or None
            name: name of the datastream object. Used to provide additional info in error messages.

        """
        if ds is not None:
            if ds.is_calibrated is True:
                raise RepeatedCalibrationError(name)
            return True
        else:
            datastream_does_not_exist_warning(name, 'calibration')
            return False

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        """Downsample all datastreams by a factor.

        This applies `scipy.signal.decimate` to all datastreams and the counter of the dataset.
        See :py:meth:`NilsPodLib.datastream.Datastream.downsample` for details.

        Warnings:
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut`, `cut_counter_val`, or `downsample` was
            used before.

        Args:
            factor: Factor by which the dataset should be downsampled.
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        s = inplace_or_copy(self, inplace)
        for key, val in s.datastreams:
            setattr(s, key, val.downsample(factor))
        s.counter = decimate(s.counter, factor, axis=0)
        return s

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        """Cut all datastreams of the dataset.

        This is equivalent to applying the following slicing to all datastreams and the counter: array[start:stop:step]

        Warnings:
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut` or `cut_counter_val` was used before.

        Args:
            start: Start index
            stop: Stop index
            step: Step size of the cut
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        s = inplace_or_copy(self, inplace)

        for key, val in s.datastreams:
            setattr(s, key, val.cut(start, stop, step))
        s.counter = s.counter[start:stop:step]
        return s

    def cut_counter_val(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
                        inplace: bool = False) -> T:
        """Cut the dataset based on values in the counter and not the index.

        Instead of just cutting the datastream based on its index, it is cut based on the counter value.
        This is equivalent to applying the following pandas style slicing to all datastreams and the counter:
        array.loc[start:stop:step]

        Warnings:
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut` or `cut_counter_val` was used before.

        Notes:
            The method searches the respective index for the start and the stop value in the `counter` and calls `cut`
            with these values.
            The step size will be passed directly and not modified (i.e. the step size will not respect downsampling or
            similar operations done beforehand).

        Args:
            start: Start value in counter
            stop: Stop value in counter
            step: Step size of the cut
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        """
        if start:
            start = np.searchsorted(self.counter, start)
        if stop:
            stop = np.searchsorted(self.counter, stop)
        return self.cut(start, stop, step, inplace=inplace)

    def cut_to_syncregion(self: Type[T], end: bool = False, warn_thres: Optional[int] = 30, inplace: bool = False) -> T:
        """Cut the dataset to the region indicated by the first and last sync package received from master.

        This cuts the dataset to the values indicated by `info.sync_index_start` and `info.sync_index_stop`.
        In case the dataset was a sync-master (`info.sync_role = 'master'`) this will have no effect and the dataset
        will be returned unmodified.

        Notes:
            Usually to work with multiple syncronised datasets, a `SyncedSession` should be used instead of cutting
            the datasets manually. `SyncedSession.cut_to_syncregion` will cover multiple edge cases involving multiple
            datasets, which can not be handled by this method.

        Warnings:
            This function should not be used after any other methods that can modify the counter (e.g. `cut` or
            `downsample`).

            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.

        Args:
            end: Whether the dataset should be cut at the `info.last_sync_index`. Usually it can be assumed that the
                data will be synchronous for multiple seconds after the last sync package. Therefore, it might be
                acceptable to just ignore the last syncpackage and just cut the start of the dataset.
            warn_thres: Threshold in seconds from the end of a dataset. If the last syncpackage occurred more than
                warn_thres before the end of the dataset, a warning is emitted. Use warn_thres = None to silence.
                This is not relevant if the end of the dataset is cut (e.g. `end=True`)
            inplace: If True this methods modifies the current dataset object. If False, a copy of the dataset and all
                 datastream objects is created

        Raises:
            ValueError: If the dataset does not have any sync infos

        Warns:
            If a syncpackage occurred far before the last sample in the dataset. See arg `warn_thres`.

        """
        if self.info.is_synchronised is False:
            raise ValueError('Only synchronised Datasets can be cut to the syncregion')
        if self.info.sync_role == 'master':
            return inplace_or_copy(self, inplace)
        if warn_thres is not None and end is not True and self._check_sync_packages(warn_thres) is False:
            warnings.warn('The last sync package occured more than {} s before the end of the measurement.'
                          'The last region of the data should not be trusted.'.format(warn_thres))
        end = self.info.sync_index_stop if end is True else None
        return self.cut(self.info.sync_index_start, end, inplace=inplace)

    def data_as_df(self, datastreams: Optional[Sequence[str]] = None, index: Optional[str] = None) -> pd.DataFrame:
        """Export the datastreams of the dataset in a single pandas DataFrame.

        Args:
            datastreams: Optional list of datastream names, if only specific ones should be included. Datastreams that
                are not part of the current dataset will be silently ignored.
            index: Specify which index should be used for the dataset. The options are:
                "counter": For the actual counter
                "time": For the time in seconds since the first sample
                "utc": For the utc time stamp of each sample
                "utc_datetime": for a pandas DateTime index in UTC time
                None: For a simple index (0...N)

        Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Raises:
            ValueError: If any other than the allowed `index` values are used.

        """
        index_names = {None: 'n_samples', 'counter': 'n_samples', 'time': 't', 'utc': 'utc', 'utc_datetime': 'date'}
        if index and index not in index_names.keys():
            raise ValueError(
                'Supplied value for index ({}) is not allowed. Allowed values: {}'.format(index, index_names.keys()))

        index_name = index_names[index]

        datastreams = datastreams or self.active_sensors
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

    def imu_data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        """Export the acc and gyro datastreams of the dataset in a single pandas DataFrame.

        See Also:
            :py:meth:`NilsPodLib.dataset.Dataset.data_as_df`

        Args:
            index: Specify which index should be used for the dataset. The options are:
                "counter": For the actual counter
                "time": For the time in seconds since the first sample
                "utc": For the utc time stamp of each sample
                "utc_datetime": for a pandas DateTime index in UTC time
                None: For a simple index (0...N)

        Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Raises:
            ValueError: If any other than the allowed `index` values are used.

        """
        return self.data_as_df(datastreams=['acc', 'gyro'], index=index)

    def find_closest_calibration(self, folder: path_t, recursive: bool = False, filter_cal_type: Optional[str] = None,
                                 before_after: Optional[str] = None) -> Path:
        """Find the closest calibration info to the start of the measurement.

        As this only checks the filenames, this might return a false positive depending on your folder structure and
        naming.

        Args:
            folder: Basepath of the folder to search
            recursive: If the folder should be searched recursive or not.
            filter_cal_type: Whether only files obtain with a certain calibration type should be found.
                This will look for the `CalType` inside the json file and hence cause performance problems.
                If None, all found files will be returned.
                For possible values, see the `imucal` library.
            before_after: Can either be 'before' or 'after', if the search should be limited to calibrations that were
                either before or after the specified date.

        See Also:
            :py:func:`NilsPodLib.calibration_utils.find_calibrations_for_sensor`
            :py:func:`NilsPodLib.calibration_utils.find_closest_calibration_to_date`

        """
        # TODO: Test
        # TODO: Make folder path optional once there is a way to get default calibrations
        return find_closest_calibration_to_date(
            sensor_id=self.info.sensor_id,
            cal_time=self.info.utc_datetime_start,
            folder=folder,
            recursive=recursive,
            filter_cal_type=filter_cal_type,
            before_after=before_after
        )

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


def parse_binary(path: path_t, legacy_error: bool = True) -> Tuple[Dict[str, np.ndarray],
                                                                   np.ndarray,
                                                                   Header]:
    """Parse a binary NilsPod session file and read the header and the data.

    Args:
        path: Path to the file
        legacy_error: The method checks, if the binary file was created with a compatible Firmware Version.
            If `legacy_error` is True, this will raise an error, if an unsupported version is detected.
            If `legacy_error` is False, only a warning will be displayed.

    Returns:
        The sensor data as dictionary
        The counter values
        The session header

    Raises:
        VersionError: If unsupported FirmwareVersion is detected and `legacy_error` is True

    """
    header_bytes, data_bytes = get_header_and_data_bytes(path)

    version = get_strict_version_from_header_bytes(header_bytes)
    legacy_support_check(version, as_warning=not legacy_error)

    session_header = Header.from_bin_array(header_bytes[1:])

    sample_size = session_header.sample_size
    n_samples = session_header.n_samples

    data = read_binary_uint8(data_bytes, sample_size, n_samples)
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
