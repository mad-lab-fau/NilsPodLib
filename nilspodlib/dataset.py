# -*- coding: utf-8 -*-
"""Dataset represents a measurement session of a single sensor_type."""
import datetime
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from packaging.version import Version
from typing_extensions import Self

from nilspodlib.calibration_utils import (
    find_calibrations_for_sensor,
    find_closest_calibration_to_date,
    load_and_check_cal_info,
)
from nilspodlib.consts import GRAV, SENSOR_SAMPLE_LENGTH
from nilspodlib.datastream import Datastream
from nilspodlib.exceptions import (
    InvalidInputFileError,
    LegacyWarning,
    RepeatedCalibrationError,
    SynchronisationWarning,
    datastream_does_not_exist_warning,
)
from nilspodlib.header import Header
from nilspodlib.legacy import find_conversion_function, legacy_support_check
from nilspodlib.utils import (
    convert_little_endian,
    get_header_and_data_bytes,
    get_strict_version_from_header_bytes,
    inplace_or_copy,
    path_t,
    raise_timezone_error,
    read_binary_uint8,
)

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401


class Dataset:  # noqa: too-many-public-methods
    """Class representing a logged session of a single NilsPod.

    .. warning:: Some operations on the dataset should not be performed after each other, as they can lead to unexpected
                 results.
                 The respective methods have specific warnings in their docstring.

    Each instance has 3 important (groups of attributes):

    - self.info: A instance of `nilspodlib.header.Header` containing all the meta info about the measurement.
    - self.counter: The continuous counter created by the sensor.
      It is in particular important to synchronise multiple datasets that were recorded at the same time
      (see `nilspodlib.session.SyncedSession`).
    - datastream: The actual sensor_type data accessed directly by the name of the sensor_type
      (e.g. acc, gyro, baro, ...).
      Each sensor_type data is wrapped in a `NilPodLib.datastream.Datastream` object.

    Attributes
    ----------
    path :
        Path pointing to the recording file (if dataset was loaded from a file)
    info :
        Metadata of the recording
    size :
        The number of samples in the dataset.
    counter :
        The continuous counter of the sensor.
    time_counter :
        Counter in seconds since first sample.
    utc_counter :
        Counter as utc timestamps.
    utc_datetime_counter :
        Counter as np.datetime64 in UTC timezone.
    active_sensor :
        The enabled sensors in the dataset.
    datastreams :
        Iterator over all available datastreams/sensors
    acc :
        Optional accelerometer datastream.
    gyro :
        Optional gyroscope datastream.
    mag :
        Optional magnetometer datastream.
    baro :
        Optional barometer datastream.
    analog :
        Optional analog datastream.
        Its content will depend on the exact recording and sensor used.
    ecg :
        Optional ECG datastream.
    ppg :
        Optional PPG datastream.
    temperature :
        Optional temperature reading datastream.

    """

    path: path_t
    acc: Optional["Datastream"] = None
    gyro: Optional["Datastream"] = None
    mag: Optional["Datastream"] = None
    baro: Optional["Datastream"] = None
    analog: Optional["Datastream"] = None
    ecg: Optional["Datastream"] = None
    ppg: Optional["Datastream"] = None
    temperature: Optional["Datastream"] = None
    counter: np.ndarray
    info: Header

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
    def utc_datetime_counter(self) -> pd.Series:
        """Counter as pandas datetime series in UTC timezone."""
        return pd.Series((self.utc_counter * 1e6).astype("datetime64[us]")).dt.tz_localize("UTC")

    @property
    def local_datetime_counter(self) -> pd.Series:
        """Counter as pandas datetime series in the specified timezone."""
        raise_timezone_error(self.info.timezone)
        return self.utc_datetime_counter.dt.tz_convert(self.info.timezone)

    @property
    def time_counter(self) -> np.ndarray:
        """Counter in seconds since first sample."""
        return (self.counter - self.counter[0]) / self.info.sampling_rate_hz

    def __init__(self, sensor_data: Dict[str, np.ndarray], counter: np.ndarray, info: Header):
        """Get new Dataset instance.

        .. note::
            Usually you shouldn't use this init directly.
            Use the provided `from_bin_file` constructor to handle loading recorded NilsPod Sessions.

        Parameters
        ----------
        sensor_data :
            Dictionary with name of sensor_type and sensor_type data as np.array
            The data needs to be 2D with time/counter as first dimension
        counter :
            The counter created by the sensor_type. Should have the same length as all datasets
        info :
            Header instance containing all Metainfo about the measurement.

        """
        self.counter = counter
        self.info = info

        calibration_dict = {
            "acc": self._factory_calibrate_acc,
            "gyro": self._factory_calibrate_gyro,
            "baro": self._factory_calibrate_baro,
            "temperature": self._factory_calibrate_temperature,
        }

        for k, v in sensor_data.items():
            ds = Datastream(v, self.info.sampling_rate_hz, sensor_type=k)
            # Apply factory calibration:
            if k in calibration_dict:
                ds = calibration_dict[k](ds)
            setattr(self, k, ds)

    @classmethod
    def from_bin_file(
        cls,
        path: path_t,
        *,
        legacy_support: str = "error",
        force_version: Optional[Version] = None,
        tz: Optional[str] = None,
    ) -> Self:
        """Create a new Dataset from a valid .bin file.

        Parameters
        ----------
        path :
            Path to the file
        legacy_support :
            This indicates how to deal with old firmware versions.
            If `error`: An error is raised, if an unsupported version is detected.
            If `warn`: A warning is raised, but the file is parsed without modification
            If `resolve`: A legacy conversion is performed to load old files. If no suitable conversion is found,
            an error is raised. See the `legacy` package and the README to learn more about available
            conversions.
        force_version
            Instead of relying on the version provided in the session header, the legacy support will be determined
            based on the version provided here.
            This is only used, if `legacy_support="resolve"`.
            This option can be helpful, when testing with development firmware images that don't have official version
            numbers.
        tz
            Optional timezone str of the recording.
            This can be used to localize the start and end time.
            Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
            recording.

        Raises
        ------
        VersionError
            If unsupported FirmwareVersion is detected and `legacy_error` is True

        """
        path = Path(path)
        if path.suffix != ".bin":
            ValueError('Invalid file type! Only ".bin" files are supported not {}'.format(path))

        sensor_data, counter, info = parse_binary(
            path, legacy_support=legacy_support, force_version=force_version, tz=tz
        )
        s = cls(sensor_data, counter, info)

        s.path = path
        return s

    def calibrate_imu(self, calibration: Union["CalibrationInfo", path_t], inplace: bool = False) -> Self:
        """Apply a calibration to the Acc and Gyro datastreams.

        The final units of the datastreams will depend on the used calibration values, but must likely they will be "g"
        for the Acc and "dps" (degrees per second) for the Gyro.

        Parameters
        ----------
        calibration :
            calibration object or path to .json file, that can be used to create one.
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created
            Notes:
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created
            Notes:
            This just combines `calibrate_acc` and `calibrate_gyro`.

        """
        s = inplace_or_copy(self, inplace)
        check = [self._check_calibration(s.acc, "acc"), self._check_calibration(s.gyro, "gyro")]
        if all(check):
            calibration = load_and_check_cal_info(calibration)
            acc, gyro = calibration.calibrate(s.acc.data, s.gyro.data, acc_unit=s.acc.unit, gyr_unit=s.gyro.unit)
            s.acc.data = acc
            s.gyro.data = gyro
            s.acc.is_calibrated = True
            s.acc.calibrated_unit = calibration.acc_unit
            s.gyro.is_calibrated = True
            s.gyro.calibrated_unit = calibration.gyr_unit
        return s

    def _factory_calibrate_gyro(self, gyro: Datastream) -> Datastream:
        """Apply a factory calibration to the Gyro datastream.

        The values used for that are taken from the datasheet of the sensor_type and are likely not to be accurate.
        For any tasks requiring precise sensor_type outputs, `calibrate_gyro` should be used with measured calibration
        values.

        The final units of the output will be "deg/s" (degrees per second) for the Gyro.

        Parameters
        ----------
        gyro :
            The uncalibrated gyro Datastream

        """
        assert gyro.sensor_type == "gyro"
        if self._check_calibration(gyro, "gyro", factory=True) is True:
            gyro.data /= 2**16 / self.info.gyro_range_dps / 2
            gyro.is_factory_calibrated = True
        return gyro

    def _factory_calibrate_acc(self, acc: Datastream) -> Datastream:
        """Apply a factory calibration to the Acc datastream.

        The values used for that are taken from the datasheet of the sensor_type and are likely not to be accurate.
        For any tasks requiring precise sensor_type outputs, `calibrate_acc` should be used with measured calibration
        values.

        The final units of the output will be "m/s^2" for the Acc.

        Parameters
        ----------
        acc :
            The uncalibrated acc Datastream

        """
        assert acc.sensor_type == "acc"
        if self._check_calibration(acc, "acc", factory=True) is True:
            acc.data /= 2**16 / self.info.acc_range_g / 2 / GRAV
            acc.is_factory_calibrated = True
        return acc

    def _factory_calibrate_baro(self, baro: Datastream) -> Datastream:
        """Apply a calibration to the Baro datastream.

        The values used for that are taken from the datasheet of the sensor_type and are likely not to be accurate.
        In general, if baro measurements are used to estimate elevation, the estimation should be calibrated relative to
        a reference altitude.

        The final units of the output will be "millibar" (equivalent to Hectopacal) for the Baro.

        Parameters
        ----------
        baro :
            The uncalibrated baro Datastream

        """
        assert baro.sensor_type == "baro"
        if self._check_calibration(baro, "baro", factory=True) is True:
            baro.data = (baro.data + 101325) / 100.0
            baro.is_factory_calibrated = True
        return baro

    def _factory_calibrate_temperature(self, temperature: Datastream) -> Datastream:
        """Apply a factory calibration to the temperature datastream.

        The values used for that are taken from the datasheet of the sensor_type

        The final unit is Celsius.

        Parameters
        ----------
        temperature :
            The uncalibrated baro temperature


        """
        assert temperature.sensor_type == "temperature"
        if self._check_calibration(temperature, "temperature", factory=True) is True:
            temperature.data = temperature.data * (2**-9) + 23
            temperature.is_factory_calibrated = True
        return temperature

    @staticmethod
    def _check_calibration(ds: Optional[Datastream], name: str, factory: bool = False):
        """Check if a specific datastream is already marked as calibrated, or if the datastream does not exist.

        In case the datastream is already calibrated a `RepeatedCalibrationError` is raised.
        In case the datastream does not exist, a warning is raised.

        Parameters
        ----------
        ds :
            datastream object or None
        name :
            name of the datastream object. Used to provide additional info in error messages.
        factory :
            If we want to check for factory calibration or not.
            If True, it will only be checked if the dataset is factory calibrated.
            If False, it will be checked if the dataset is normally calibrated.

        """
        if ds is not None:
            if factory is True:
                check_val = ds.is_factory_calibrated
            else:
                check_val = ds.is_calibrated
            if check_val is True:
                raise RepeatedCalibrationError(name, factory)
            return True
        datastream_does_not_exist_warning(name, "calibration")
        return False

    def downsample(self, factor: int, inplace: bool = False) -> Self:
        """Downsample all datastreams by a factor.

        This applies `scipy.signal.decimate` to all datastreams and the counter of the dataset.
        See :py:meth:`nilspodlib.datastream.Datastream.downsample` for details.

        .. warning::
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut`, `cut_counter_val`, or `downsample` was
            used before.

        Parameters
        ----------
        factor :
            Factor by which the dataset should be downsampled.
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created

        """
        from scipy.signal import resample  # noqa: import-outside-toplevel

        s = inplace_or_copy(self, inplace)
        for key, val in s.datastreams:
            setattr(s, key, val.downsample(factor))
        s.counter = resample(s.counter, len(s.counter) // factor, axis=0)
        return s

    def cut(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        inplace: bool = False,
    ) -> Self:
        """Cut all datastreams of the dataset.

        This is equivalent to applying the following slicing to all datastreams and the counter: array[start:stop:step]

        .. warning ::
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut` or `cut_counter_val` was used before.

        Parameters
        ----------
        start :
            Start index
        stop :
            Stop index
        step :
            Step size of the cut
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created

        """
        s = inplace_or_copy(self, inplace)

        for key, val in s.datastreams:
            setattr(s, key, val.cut(start, stop, step))
        sl = slice(start, stop, step)
        s.counter = s.counter[sl]
        return s

    def cut_counter_val(
        self,
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        inplace: bool = False,
    ) -> Self:
        """Cut the dataset based on values in the counter and not the index.

        Instead of just cutting the datastream based on its index, it is cut based on the counter value.
        This is equivalent to applying the following pandas style slicing to all datastreams and the counter:
        array.loc[start:stop:step]

        .. warning::
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.
            For example `cut_to_syncregion` will not work correctly, if `cut` or `cut_counter_val` was used before.

        Notes
        -----
        The method searches the respective index for the start and the stop value in the `counter` and calls `cut`
        with these values.
        The step size will be passed directly and not modified (i.e. the step size will not respect downsampling or
        similar operations done beforehand).

        Parameters
        ----------
        start :
            Start value in counter
        stop :
            Stop value in counter
        step :
            Step size of the cut
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created

        """
        if start:
            if start < self.counter[0]:
                raise ValueError("{} out of bounds for counter starting at {}".format(start, self.counter))
            start = np.searchsorted(self.counter, start)
        if stop:
            if stop > self.counter[-1]:
                raise ValueError("{} out of bounds for counter ending at {}".format(start, self.counter))
            stop = np.searchsorted(self.counter, stop)
        return self.cut(start, stop, step, inplace=inplace)

    def cut_to_syncregion(
        self, start: bool = True, end: bool = False, warn_thres: Optional[int] = 30, inplace: bool = False
    ) -> Self:
        """Cut the dataset to the region indicated by the first and last sync package received from master.

        This cuts the dataset to the values indicated by `info.sync_index_start` and `info.sync_index_stop`.
        In case the dataset was a sync-master (`info.sync_role = 'master'`) this will have no effect and the dataset
        will be returned unmodified.

        .. warning::
            This function should not be used after any other methods that can modify the counter (e.g. `cut` or
            `downsample`).

        .. warning::
            This will not modify any values in the header/info the dataset. I.e. the number of samples in the header/
            sync index values. Using methods that rely on these values might result in unexpected behaviour.

        Notes
        -----
        Usually to work with multiple syncronised datasets, a `SyncedSession` should be used instead of cutting
        the datasets manually. `SyncedSession.cut_to_syncregion` will cover multiple edge cases involving multiple
        datasets, which can not be handled by this method.


        Parameters
        ----------
        start :
            Whether the dataset should be cut at the `info.sync_index_start`.
            If this is False, a jump in the counter will remain.
            The only usecase for not cutting at the start is when the counters are already perfectly aligned.
        end :
            Whether the dataset should be cut at the `info.sync_index_stop`. Usually it can be assumed that the
            data will be synchronous for multiple seconds after the last sync package. Therefore, it might be
            acceptable to just ignore the last syncpackage and just cut the start of the dataset.
        warn_thres :
            Threshold in seconds from the end of a dataset. If the last syncpackage occurred more than
            warn_thres before the end of the dataset, a warning is emitted. Use warn_thres = None to silence.
            This is not relevant if the end of the dataset is cut (e.g. `end=True`)
        inplace :
            If True this methods modifies the current dataset object. If False, a copy of the dataset and all
            datastream objects is created

        Raises
        ------
        ValueError
            If the dataset does not have any sync infos
        ValueError
            If the dataset does not have any sync infos

        Warnings
        --------
        UserWarning
            If a syncpackage occurred far before the last sample in the dataset. See arg `warn_thres`.

        """
        if self.info.is_synchronised is False:
            raise ValueError("Only synchronised Datasets can be cut to the syncregion")
        if self.info.sync_role == "master":
            return inplace_or_copy(self, inplace)
        if warn_thres is not None and end is not True and self._check_sync_packages(warn_thres) is False:
            warnings.warn(
                "The last sync package occured more than {} s before the end of the measurement."
                "The last region of the data should not be trusted.".format(warn_thres),
                SynchronisationWarning,
            )
        end = self.info.sync_index_stop if end is True else None
        start = self.info.sync_index_start if start is True else None
        return self.cut(start, end, inplace=inplace)

    def data_as_df(
        self,
        datastreams: Optional[Sequence[str]] = None,
        index: Optional[str] = None,
        include_units: Optional[bool] = False,
    ) -> pd.DataFrame:
        """Export the datastreams of the dataset in a single pandas DataFrame.

        Parameters
        ----------
        datastreams :
            Optional list of datastream names, if only specific ones should be included. Datastreams that
            are not part of the current dataset will be silently ignored.
        index :
            Specify which index should be used for the dataset. The options are:
            "counter": For the actual counter
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`

        Notes
        -----
        This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        Therefore, it will use the column information of each datastream.

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        index_names = {
            None: "n_samples",
            "counter": "n_samples",
            "time": "t",
            "utc": "utc",
            "utc_datetime": "date",
            "local_datetime": "date ({})".format(self.info.timezone),
        }
        if index and index not in index_names:
            raise ValueError(f"Supplied value for index ({index}) is not allowed. Allowed values: {index_names.keys()}")

        index_name = index_names[index]

        datastreams = datastreams or self.active_sensors
        dfs = [s.data_as_df(include_units=include_units) for k, s in self.datastreams if k in datastreams]

        df = pd.concat(dfs, axis=1)

        if index:
            if index != "counter":
                index += "_counter"
            index = getattr(self, index, None)
            df.index = index
        else:
            df = df.reset_index(drop=True)
        df.index.name = index_name
        return df

    def imu_data_as_df(self, index: Optional[str] = None, include_units: Optional[bool] = False) -> pd.DataFrame:
        """Export the acc and gyro datastreams of the dataset in a single pandas DataFrame.

        See Also
        --------
        nilspodlib.dataset.Dataset.data_as_df

        Parameters
        ----------
        index :
            Specify which index should be used for the dataset. The options are:
            "counter": For the actual counter
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`

        Notes
        -----
        This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        Therefore, it will use the column information of each datastream.


        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        return self.data_as_df(datastreams=["acc", "gyro"], index=index, include_units=include_units)

    def find_calibrations(
        self,
        folder: Optional[path_t] = None,
        recursive: bool = True,
        filter_cal_type: Optional[str] = None,
        ignore_file_not_found: Optional[bool] = False,
    ) -> List[Path]:
        """Find all calibration infos that belong to a given sensor_type.

        As this only checks the filenames, this might return a false positive depending on your folder structure and
        naming.

        Parameters
        ----------
        folder :
            Basepath of the folder to search. If None, tries to find a default calibration
        recursive :
            If the folder should be searched recursive or not.
        filter_cal_type :
            Whether only files obtain with a certain calibration type should be found.
            This will look for the `CalType` inside the json file and hence cause performance problems.
            If None, all found files will be returned.
            For possible values, see the `imucal` library.
        ignore_file_not_found :
            If True this function will not raise an error, but rather return an empty list, if no
            calibration files were found for the specific sensor_type.

        See Also
        --------
        nilspodlib.calibration_utils.find_calibrations_for_sensor

        """
        # TODO: Test
        return find_calibrations_for_sensor(
            sensor_id=self.info.sensor_id,
            folder=folder,
            recursive=recursive,
            filter_cal_type=filter_cal_type,
            ignore_file_not_found=ignore_file_not_found,
        )

    def find_closest_calibration(
        self,
        folder: Optional[path_t] = None,
        recursive: bool = True,
        filter_cal_type: Optional[str] = None,
        before_after: Optional[str] = None,
        warn_thres: datetime.timedelta = datetime.timedelta(days=30),  # noqa E252
        ignore_file_not_found: Optional[bool] = False,
    ) -> Path:
        """Find the closest calibration info to the start of the measurement.

        As this only checks the filenames, this might return a false positive depending on your folder structure and
        naming.

        Parameters
        ----------
        folder :
            Basepath of the folder to search. If None, tries to find a default calibration
        recursive :
            If the folder should be searched recursive or not.
        filter_cal_type :
            Whether only files obtain with a certain calibration type should be found.
            This will look for the `CalType` inside the json file and hence cause performance problems.
            If None, all found files will be returned.
            For possible values, see the `imucal` library.
        before_after :
            Can either be 'before' or 'after', if the search should be limited to calibrations that were
            either before or after the specified date.
        warn_thres :
            If the distance to the closest calibration is larger than this threshold, a warning is emitted
        ignore_file_not_found :
            If True this function will not raise an error, but rather return `None`, if no
            calibration files were found for the specific sensor_type.

        See Also
        --------
        nilspodlib.calibration_utils.find_calibrations_for_sensor
        nilspodlib.calibration_utils.find_closest_calibration_to_date

        """
        # TODO: Test
        return find_closest_calibration_to_date(
            sensor_id=self.info.sensor_id,
            cal_time=self.info.utc_datetime_start,
            folder=folder,
            recursive=recursive,
            filter_cal_type=filter_cal_type,
            before_after=before_after,
            warn_thres=warn_thres,
            ignore_file_not_found=ignore_file_not_found,
        )

    def _check_sync_packages(self, threshold_s: int = 30, where="end") -> bool:
        """Check if the last sync package occurred far from the actual end of the recording.

        This can be the case, if the master stopped sending packages, or if the sensor could not receive any new sync
        info for a prelonged period of time.
        In particular in the latter case, careful review of the data is advised.
        """
        if self.info.sync_role == "slave":
            if where == "end":
                tmp = len(self.counter) - self.info.sync_index_stop
            elif where == "start":
                tmp = self.info.sync_index_start
            else:
                raise ValueError('Invalid value for "where" encountered.')
            if tmp > threshold_s * self.info.sampling_rate_hz:
                return False
        return True


def parse_binary(
    path: path_t, legacy_support: str = "error", force_version: Optional[Version] = None, tz: Optional[str] = None
) -> Tuple[Dict[str, np.ndarray], np.ndarray, Header]:
    """Parse a binary NilsPod session file and read the header and the data.

    Parameters
    ----------
    path :
        Path to the file
    legacy_support :
        This indicates how to deal with old firmware versions.
        If `error`, An error is raised, if an unsupported version is detected.
        If `warn`, A warning is raised, but the file is parsed without modification
        If `resolve`, A legacy conversion is performed to load old files. If no suitable conversion is found,
        an error is raised. See the `legacy` package and the README to learn more about available conversions.
    force_version
        Instead of relying on the version provided in the session header, the legacy support will be determined based on
        the version provided here.
        This is only used, if `legacy_support="resolve"`.
        This option can be helpful, when testing with development firmware images that don't have official version
        numbers.
    tz
        Optional timezone str of the recording.
        This can be used to localize the start and end time.
        Note, this should not be the timezone of your current PC, but the timezone relevant for the specific
        recording.

    Returns
    -------
    sensor_data :
        The sensor data as dictionary
    counter :
        The counter values
    session_header :
        The session header

    Raises
    ------
    VersionError
        If unsupported FirmwareVersion is detected and `legacy_error` is `error`
    VersionError
        If `legacy_error` is `resolve`, but no suitable conversion is found.

    """
    header_bytes, data_bytes = get_header_and_data_bytes(path)

    version = get_strict_version_from_header_bytes(header_bytes)

    if legacy_support == "resolve":
        version = force_version or version
        header_bytes, data_bytes = find_conversion_function(version, in_memory=True)(header_bytes, data_bytes)
    elif legacy_support in ["error", "warn"]:
        legacy_support_check(version, as_warning=(legacy_support == "warn"))
    else:
        raise ValueError("legacy_support must be one of 'resolve', 'error' or 'warn'")

    session_header = Header.from_bin_array(header_bytes[1:], tz=tz)

    sample_size = session_header.sample_size
    n_samples = session_header.n_samples

    data = read_binary_uint8(data_bytes, sample_size, n_samples)

    counter, sensor_data = split_into_sensor_data(data, session_header)

    if session_header.strict_version_firmware >= Version("0.13.0") and len(counter) != session_header.n_samples:
        warnings.warn(
            "The number of samples in the dataset does not match the number indicated by the header. "
            "This might indicate a corrupted file",
            LegacyWarning,
        )

    return sensor_data, counter, session_header


def split_into_sensor_data(data: np.ndarray, session_header: Header) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """Split/Parse the binary data into the different sensors and the counter."""
    sensor_data = {}

    idx = 0
    for sensor in session_header.enabled_sensors:
        bits, channel, dtype = SENSOR_SAMPLE_LENGTH[sensor]
        bits_per_channel = bits // channel
        tmp = np.full((len(data), channel), np.nan)
        for i in range(channel):
            tmp[:, i] = convert_little_endian(
                np.atleast_2d(data[:, idx : idx + bits_per_channel]).T, dtype=dtype
            ).astype(float)
            idx += bits_per_channel
        sensor_data[sensor] = tmp

    len_counter, _, counter_dtype = SENSOR_SAMPLE_LENGTH["counter"]

    # Sanity Check:
    if not idx + len_counter == data.shape[-1]:
        expected_cols = idx
        all_cols = data.shape[-1] - len_counter
        raise InvalidInputFileError(
            f"The input file has an invalid format. {expected_cols} data columns expected based on the header, "
            f"but {all_cols} exist."
        )

    counter = convert_little_endian(np.atleast_2d(data[:, -len_counter:]).T, dtype=counter_dtype)

    return counter, sensor_data
