"""Session groups multiple Datasets from sensors recorded at the same time."""
import datetime
import warnings
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Union

import numpy as np
from packaging.version import Version
from typing_extensions import Self

from nilspodlib._session_base import _MultiDataset
from nilspodlib.dataset import Dataset
from nilspodlib.exceptions import SessionValidationError, SynchronisationError, SynchronisationWarning
from nilspodlib.header import _ProxyHeader
from nilspodlib.utils import convert_to_local_time, inplace_or_copy, path_t, validate_existing_overlap

if TYPE_CHECKING:
    import pandas as pd
    from imucal import CalibrationInfo


_SYNC_DEBUGGING_TIPS = (
    "Manually plot all counters, if you can spot potential issues. "
    "You should see exactly one large jump in the counter value close to the reported `sync_index_start` from the "
    "header.\n"
    "If you see multiple jumps, the clock of one of the sensors might be broken, or data was not stored/transferred "
    "correctly. "
    "In the latter case, try to re-download the session from the sensor. "
    "If the issue persists, you might be out of luck (unless... keep reading ;) ).\n"
    "If only the last couple of counter values seem broken, you might have run into a known (but unsolved bug) with "
    "the NilsPod Firmware. "
    "In this case, cut the last affected samples and retry the synchronisation:\n\n"
    ">>> session = session.cut(stop=-n_affected_samples)\n"
    ">>> session = session.align_to_syncregion()"
)


class Session(_MultiDataset):
    """Object representing a collection of Datasets.

    .. note:: By default a session makes no assumption about when and how datasets are recorded.
              It just provides an interface to manipulate multiple datasets at once.
              If you have datasets that were recorded simultaneously with active sensor_type synchronisation,
              you should use a `SyncedSession` instead of a `Session` to take full advantage of this.

    A session can access all the same attributes and most of the methods provided by a dataset.
    However, instead of returning a single value/acting only on a single dataset, it will return a tuple of values (one
    for each dataset) or modify all datasets of a session.
    You can also use the `self.info` object to access header information of all datasets at the same time.
    All return values will be in the same order as `self.datasets`.

    Attributes
    ----------
    datasets
        A tuple of the datasets belonging to the session
    info
        Get the metadata (Header) information for all datasets

    See Also
    --------
    nilspodlib.session.SyncedSession

    """

    datasets: tuple[Dataset]

    @property
    def info(self):
        """Get metainfo for all datasets.

        All attributes of :py:class:`nilspodlib.header.HeaderInfo` are supported in read-only mode.
        """
        return _ProxyHeader(tuple(d.info for d in self.datasets))

    def __init__(self, datasets: Iterable[Dataset]):
        """Create new session.

        Instead of this init you can also use the factory methods :py:meth:`~nilspodlib.session.Session.from_file_paths`
        and :py:meth:`~nilspodlib.session.Session.from_folder_path`.

        Parameters
        ----------
        datasets :
            List of :py:class:`nilspodlib.dataset.Dataset` instances, which should be grouped into a session.

        """
        self.datasets = tuple(datasets)

    @classmethod
    def from_file_paths(
        cls,
        paths: Iterable[path_t],
        legacy_support: str = "error",
        force_version: Version | None = None,
        tz: str | None = None,
    ) -> Self:
        """Create a new session from a list of files pointing to valid .bin files.

        Parameters
        ----------
        paths :
            List of paths pointing to files to be included
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

        """
        ds = (
            Dataset.from_bin_file(p, legacy_support=legacy_support, force_version=force_version, tz=tz) for p in paths
        )
        return cls(ds)

    @classmethod
    def from_folder_path(
        cls,
        base_path: path_t,
        filter_pattern: str = "*.bin",
        legacy_support: str = "error",
        force_version: Version | None = None,
        tz: str | None = None,
    ) -> Self:
        """Create a new session from a folder path containing valid .bin files.

        Parameters
        ----------
        base_path :
            Path to the folder
        filter_pattern :
            regex that can be used to filter the files in the folder. This is passed to Pathlib.glob()
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

        """
        ds = list(Path(base_path).glob(filter_pattern))
        if not ds:
            raise ValueError(f'No files matching "{filter_pattern}" where found in {base_path}')
        return cls.from_file_paths(ds, legacy_support=legacy_support, force_version=force_version, tz=tz)

    def get_dataset_by_id(self, sensor_id: str) -> Dataset:
        """Get a specific dataset by its sensor_type id.

        Parameters
        ----------
        sensor_id :
            Four letter/digit unique id of the sensor

        """
        return self.datasets[self.info.sensor_id.index(sensor_id)]

    def calibrate_imu(self, calibrations: Iterable[Union["CalibrationInfo", path_t]], inplace: bool = False) -> Self:
        """Calibrate the imus of all datasets by providing a list of calibration infos.

        If you do not want to calibrate a specific IMU, you can pass `None` for its position.

        Parameters
        ----------
        calibrations :
            List of calibration infos in the same order than `self.datasets`
        inplace :
            If True this methods modifies the current session object. If False, a copy of the sesion and all
            dataset objects is created

        See Also
        --------
        nilspodlib.dataset.Dataset.calibrate_imu

        """
        s = inplace_or_copy(self, inplace)
        s.datasets = [
            d.calibrate_imu(c, inplace=True) if c else d for d, c in zip(s.datasets, calibrations, strict=False)
        ]
        return s


class SyncedSession(Session):
    """Object representing a collection of Datasets recorded with active synchronisation.

    A session can access all the same attributes and most of the methods provided by a dataset.
    However, instead of returning a single value/acting only on a single dataset, it will return a tuple of values (one
    for each dataset) or modify all datasets of a session.
    You can also use the `self.info` object to access header information of all datasets at the same time.
    All return values will be in the same order as `self.datasets`.

    To synchronise a dataset, you usually want to call :py:meth:`~nilspodlib.session.SyncedSession.cut_to_syncregion`
    on the session.
    The resulting session is considered fully synchronised (depending on the parameters chosen).
    This means that all datasets have the same length and the exact same counter.
    However, note, that the header information of the individual datasets will not be updated to reflect the sync.
    This means that header values like `number_of_samples` or the start and stop times will not match the data anymore.
    As a substitute you can use a set of direct attributes on the session (e.g. `session_utc_start`, `session_duration`,
    etc.)

    Attributes
    ----------
    datasets
        A tuple of the datasets belonging to the session
    info
        Get the metadata (Header) information for all datasets
    master
        Get the dataset belonging to the sync-server sensor.
    slaves
        Get the datasets belonging to all the sync-clients sensors.
    session_utc_start
       Start time of the session as utc timestamp.
       You need to `cut_to_sync_region` before you can obtain this value.
    session_utc_stop
       Stop time of the session as utc timestamp.
       You need to `cut_to_sync_region` before you can obtain this value.
    session_duration
        Duration of the session in seconds.
        You need to `cut_to_sync_region` before you can obtain this value.
    session_utc_datetime_start
        Start time of the session as utc datetime.
        You need to `cut_to_sync_region` before you can obtain this value.
    session_utc_datetime_stop
        Stop time of the session as utc datetime.
        You need to `cut_to_sync_region` before you can obtain this value.
    session_local_datetime_start
        The start time of the session in the timezone of the session.
        You need to `cut_to_sync_region` before you can obtain this value.
    session_local_datetime_stop
        The stop time of the session in the timezone of the session.
        You need to `cut_to_sync_region` before you can obtain this value.
    VALIDATE_ON_INIT
        If True all synced sessions will be checked on init.
        These checks include testing, if all datasets are really part of a single measurement.
        In rare cases, it might be useful to deactivate these checks and force the creation of a synced session.
        In this case you need to set this **class** attribute to false before loading the session:

        >>> SyncedSession.VALIDATE_ON_INIT = False
        >>> SyncedSession.from_folder_path('./my/path') # No validation will be performed

    See Also
    --------
    nilspodlib.session.Session

    """

    VALIDATE_ON_INIT: bool = True
    _fully_synced = False

    @property
    def master(self) -> Dataset:
        """Get the dataset belonging to the sync-server sensor."""
        return next(d for d in self.datasets if d.info.sync_role == "master")

    @property
    def slaves(self) -> tuple[Dataset]:
        """Get the datasets belonging to all the sync-clients sensors."""
        return tuple(d for d in self.datasets if d.info.sync_role == "slave")

    @property
    def session_utc_start(self) -> float:
        """Start time of the session as utc timestamp."""
        return self.session_utc_datetime_start.timestamp()

    @property
    def session_utc_stop(self) -> float:
        """Stop time of the session as utc timestamp."""
        return self.session_utc_datetime_stop.timestamp()

    @property
    def session_duration(self) -> float:
        """Duration of the session in seconds."""
        return self.session_utc_stop - self.session_utc_start

    @property
    def session_utc_datetime_start(self) -> datetime.datetime:
        """Start time of the session as utc datetime."""
        if not self._fully_synced:
            raise SynchronisationError("Only fully synced datasets, have valid start and stop times.")
        return self.master.info.utc_datetime_start_day_midnight + datetime.timedelta(
            seconds=self.master.counter[0] / self.master.info.sampling_rate_hz
        )

    @property
    def session_utc_datetime_stop(self) -> datetime.datetime:
        """Stop time of the session as utc datetime."""
        if not self._fully_synced:
            raise SynchronisationError("Only fully synced datasets, have valid start and stop times.")
        return self.master.info.utc_datetime_start_day_midnight + datetime.timedelta(
            seconds=self.master.counter[-1] / self.master.info.sampling_rate_hz
        )

    @property
    def session_local_datetime_start(self) -> datetime.datetime:
        """Start time of the session in the specified timezone of the session."""
        return convert_to_local_time(self.session_utc_datetime_start, self.master.info.timezone)

    @property
    def session_local_datetime_stop(self) -> datetime.datetime:
        """Stop time of the session in the specified timezone of the session."""
        return convert_to_local_time(self.session_utc_datetime_stop, self.master.info.timezone)

    def __init__(self, datasets: Iterable[Dataset]):
        """Create new synced session.

        Instead of this init you can also use the factory methods
        :py:meth:`~nilspodlib.session.SyncedSession.from_file_paths` and
        :py:meth:`~nilspodlib.session.SyncedSession.from_folder_path`.

        This init performs basic validation on the datasets.
        See :py:meth:`~nilspodlib.session.SyncedSession.validate` for details.

        Parameters
        ----------
        datasets :
            List of :py:class:`nilspodlib.dataset.Dataset` instances, which should be grouped into a session.

        """
        super().__init__(datasets)
        if self.VALIDATE_ON_INIT:
            self.validate()

    def validate(self) -> None:
        """Check if basic properties of a synced session are fulfilled.

        Raises
        ------
        ValueError
            This raises a ValueError in the following cases:
            - One or more of the datasets are not part of the same syncgroup/same sync channel
            - Multiple datasets are marked as "master"
            - One or more datasets indicate that they are not synchronised
            - One or more dataset has a different sampling rate than the others
            - If the recording times of provided datasets do not have any overlap

        """
        if not self._validate_sync_groups():
            raise SessionValidationError("The provided datasets are not part of the same sync_group.", type(self))
        master_valid, slaves_valid = self._validate_sync_role()
        if not master_valid:
            raise SessionValidationError("SyncedSessions require exactly 1 master.", type(self))
        if not slaves_valid:
            raise SessionValidationError(
                "One of the provided sessions is not correctly set to either slave or master.", type(self)
            )
        if not self._validate_sampling_rate():
            raise SessionValidationError("All provided sessions need to have the same sampling rate.", type(self))
        if not self._validate_overlapping_record_time():
            raise SessionValidationError(
                "The provided datasets do not have any overlapping time period (based on the header). , "
                "Double check, that the datasets/files you selected actually belong to the same recording. "
                "If they do, and you are still seeing this error, the clock of one of the sensors was set "
                "incorrectly.\n"
                "This is not a deal-breaker and the session can likely still be synced correctly. "
                "Load the session without validation (see below) and attempt the synchronisation. "
                "Double-check the results! "
                "If everything else worked correctly, this should still work.",
                type(self),
            )

    def _validate_sync_groups(self) -> bool:
        """Check that all _headers belong to the same sync group."""
        sync_channel = set(self.info.sync_channel)
        sync_address = set(self.info.sync_address)
        return all(len(i) == 1 for i in [sync_channel, sync_address])

    def _validate_sync_role(self) -> tuple[bool, bool]:
        """Check that there is only 1 master and all other sensors were configured as slaves."""
        roles = self.info.sync_role
        master_valid = len([i for i in roles if i == "master"]) == 1
        slaves_valid = len([i for i in roles if i == "slave"]) == len(roles) - 1
        return master_valid, slaves_valid

    def _validate_sampling_rate(self) -> bool:
        """Check that all sensors had the same sampling rate."""
        sr = set(self.info.sampling_rate_hz)
        return len(sr) == 1

    def _validate_overlapping_record_time(self) -> bool:
        """Check if all provided sessions have overlapping recording times."""
        start_times = np.array(self.info.utc_start)
        stop_times = np.array(self.info.utc_stop)
        return validate_existing_overlap(start_times, stop_times)

    def align_to_syncregion(
        self,
        cut_start: bool = False,
        cut_end: bool = False,
        inplace: bool = False,
        warn_thres: int | None = 30,
    ) -> Self:
        """Align all datasets based on regions where they were synchronised to the master.

        At the end all datasets are cut to the same length, so that the maximum overlap between all datasets is
        preserved.

        Parameters
        ----------
        cut_start :
            Whether the dataset should be cut at `info.sync_index_start`.
            If `False` a new corrected counter value will be calculated for all packages before the first
            syncpackage.
            Usually it can be assumed that this extrapolation is valid for multiple seconds before the first
            package.
        cut_end :
            Whether the dataset should be cut at the `info.sync_index_stop`. Usually it can be assumed that the
            data will be synchronous for multiple seconds after the last sync package. Therefore, it might be
            acceptable to just ignore the last syncpackage and just cut the start of the dataset.
        warn_thres :
            Threshold in seconds from the end of a dataset. If the last syncpackage occurred more than
            warn_thres before the end of the dataset, a warning is emitted. Use warn_thres = None to silence.
            This is not relevant if the end of the dataset is cut (e.g. `end=True`)
        inplace :
            If operation should be performed on the current Session object, or on a copy
            Warns:
        inplace :
            If operation should be performed on the current Session object, or on a copy
            Warns:
            If a syncpackage occurred far before the last sample in any of the dataset. See arg `warn_thres`.

        """
        # Correct counter before first sync package
        s = inplace_or_copy(self, inplace)

        if s._fully_synced is True:
            raise SynchronisationError("The session is already aligned/cut to the syncregion and can not be cut again")

        if warn_thres is not None:
            sync_start_warn = [
                d.info.sensor_id for d in s.slaves if d._check_sync_packages(warn_thres, where="start") is False
            ]
            sync_end_warn = [
                d.info.sensor_id for d in s.slaves if d._check_sync_packages(warn_thres, where="end") is False
            ]
            if any(sync_end_warn) and cut_end is not True:
                warnings.warn(
                    f"For the sensors with the ids {sync_end_warn} the last syncpackage occurred more than "
                    f"{warn_thres} s before the end of the dataset. "
                    "The last section of this data should not be trusted.",
                    SynchronisationWarning,
                )
            if any(sync_start_warn) and cut_start is not True:
                warnings.warn(
                    f"For the sensors with the ids {sync_start_warn} the first syncpackage occurred more than "
                    f"{warn_thres} s after the start of the dataset. "
                    "The first section of this data should not be trusted.",
                    SynchronisationWarning,
                )

        # Correct the jump at the beginning of the sync region in the slave counter.
        # This is important because the datasets are later cut based on their first counter value
        for slave in s.slaves:
            if slave.info.sync_index_start <= 1:
                # Unlikely edge case, but let's handle it
                # We do not need to do anything in this case
                continue
            # slave.info.sync_index_start is the first sample (as in # samples) were the index is correct.
            # Therefore, the jump occurs between sample (sync_index_start - 1) and sync_index_start.
            # or equivalently between index (starting at 0) (sync_index_start - 2) and (sync_index_start - 1)
            diff = slave.counter[slave.info.sync_index_start - 1] - slave.counter[slave.info.sync_index_start - 2] - 1
            slave.counter[: slave.info.sync_index_start - 1] += diff

        # Optionally cut to the syncregion
        s = super(SyncedSession, s).cut_to_syncregion(start=cut_start, end=cut_end, inplace=True, warn_thres=None)

        start_idx = [d.counter[0] for d in s.datasets]
        stop_idx = [d.counter[-1] for d in s.datasets]
        try:
            existing_overlap = validate_existing_overlap(np.array(start_idx), np.array(stop_idx))
        except ValueError as e:
            raise SynchronisationError(
                "For one or more datasets, the last counter value after aligning the sync regions appears to occur "
                "before the start value. "
                "This might happen because the last samples were not stored correctly on the NilsPod. "
                "This a known bug. "
                "Learn more how to debug/resolve that below\n\n" + _SYNC_DEBUGGING_TIPS
            ) from e
        if existing_overlap is False:
            raise SynchronisationError(
                "The provided datasets do not have a overlapping regions based on the counter! "
                "This is concerning and likely means that you tried to sync datasets, that actually "
                "don't belong to the same recording or one of the sessions is severely corrupted.\n\n"
                + _SYNC_DEBUGGING_TIPS
            )

        s = super(SyncedSession, s).cut_counter_val(np.max(start_idx), inplace=True)
        stop_idx = [len(d.counter) for d in s.datasets]
        s = super(SyncedSession, s).cut(stop=np.min(stop_idx), inplace=True)

        # Finally set the master counter to all slaves to really ensure identical counters
        for d in s.slaves:
            d.counter = s.master.counter
        s._fully_synced = True
        return s

    def data_as_df(  # noqa: arguments-differ
        self,
        datastreams: Sequence[str] | None = None,
        index: str | None = None,
        include_units: bool | None = False,
        concat_df: bool | None = False,
    ) -> Union[tuple["pd.DataFrame"], "pd.DataFrame"]:
        """Export all datasets of the session in a list of (or a single) pandas DataFrame.

        Parameters
        ----------
        datastreams :
            Optional list of datastream names, if only specific ones should be included. Datastreams that
            are not part of the current dataset will be silently ignored.
        index :
            Specify which index should be used for each dataset. The options are:
            "counter": For the actual counter
            "time": For the time in seconds since the first sample
            "utc": For the utc time stamp of each sample
            "utc_datetime": for a pandas DateTime index in UTC time
            "local_datetime": for a pandas DateTime index in the timezone set for the session
            None: For a simple index (0...N)
        concat_df :
            If True the individual dfs from each dataset will be concatenated. This is only supported, if the
            session is properly cut to the sync region and all the datasets have the same counter.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Returns
        -------
        Session as single or multiple dataframes
            Tuple of pd.DataFrames (one for each Dataset) or a single DataFrame if `concat_df` is set to True

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        See Also
        --------
        nilspodlib.dataset.Dataset.data_as_df

        """
        import pandas as pd  # noqa: import-outside-toplevel

        dfs = super().data_as_df(datastreams, index, include_units=include_units)
        if concat_df is True:
            if not self._fully_synced:
                raise SynchronisationError("Only fully synced datasets, can be exported as a df with unified index.")
            dfs = pd.concat(dfs, axis=1, keys=self.info.sensor_id)
        return dfs

    def imu_data_as_df(  # noqa: arguments-differ
        self,
        index: str | None = None,
        include_units: bool | None = False,
        concat_df: bool | None = False,
    ) -> Union[tuple["pd.DataFrame"], "pd.DataFrame"]:
        """Export the acc and gyro datastreams of all datasets in list of (or a single) pandas DataFrame.

        See Also
        --------
        nilspodlib.session.SyncedSession.data_as_df
        nilspodlib.dataset.Dataset.data_as_df
        nilspodlib.dataset.Dataset.imu_data_as_df

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
        concat_df :
            If True the individual dfs from each dataset will be concatenated. This is only supported, if the
            session is properly cut to the sync region and all the datasets have the same counter.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
        include_units :
            If True the column names will have the unit of the datastream concatenated with an `_`
            Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Returns
        -------
        Imu data as single or multiple dataframes
            Tuple of pd.DataFrames (one for each Dataset) or a single DataFrame if `concat_df` is set to True

        Raises
        ------
        ValueError
            If any other than the allowed `index` values are used.

        """
        return self.data_as_df(
            datastreams=["acc", "gyro"],
            index=index,
            include_units=include_units,
            concat_df=concat_df,
        )
