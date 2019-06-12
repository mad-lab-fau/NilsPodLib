# -*- coding: utf-8 -*-
"""Session groups multiple Datasets from sensors recorded at the same time.

@author: Nils Roth, Arne Küderle
"""
import warnings
from pathlib import Path
from typing import Iterable, Tuple, TypeVar, Type, Any, Optional, Union, TYPE_CHECKING, Sequence

import numpy as np

from NilsPodLib.dataset import Dataset
from NilsPodLib.header import ProxyHeader
from NilsPodLib.interfaces import CascadingDatasetInterface
from NilsPodLib.utils import validate_existing_overlap, inplace_or_copy, path_t, SynchronisationError

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401

T = TypeVar('T', bound='Session')


# TODO: Create function to parse sessions from larger folder full of datasets
# def identify_sessions(folder_path: path_t, filter_pattern: str = '*') -> Sequence[Sequence[path_t]]:
#     files = Path(folder_path).glob(filter_pattern)
#     props = dict()
#     for f in files:
#         props[f] =
# STrep1: pool by syncgroup
# Step2: pool by overlapping dates


class Session(CascadingDatasetInterface):
    datasets: Tuple[Dataset]

    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = tuple(datasets)

    def get_dataset_by_id(self, sensor_id: str) -> Dataset:
        """Get a specific dataset by its sensor id."""
        # TODO: TEsting
        return self.datasets[self.info.sensor_id.index(sensor_id)]

    @classmethod
    def from_file_paths(cls: Type[T], paths: Iterable[path_t], legacy_support: str = 'error') -> T:
        """Create a new session from a list of files pointing to valid .bin files.

        Args:
            paths: List of paths pointing to files to be included
            legacy_support: This indicates how to deal with old firmware versions.
                If `error`: An error is raised, if an unsupported version is detected.
                If `warn`: A warning is raised, but the file is parsed without modification
                If `resolve`: A legacy conversion is performed to load old files. If no suitable conversion is found,
                    an error is raised. See the `legacy` package and the README to learn more about available
                    conversions.
        """
        ds = (Dataset.from_bin_file(p, legacy_support=legacy_support) for p in paths)
        return cls(ds)

    @classmethod
    def from_folder_path(cls: Type[T], base_path: path_t, filter_pattern: str = '*.bin',
                         legacy_support: str = 'error') -> T:
        """Create a new session from a folder path containing valid .bin files.

        Args:
            base_path: Path to the folder
            filter_pattern: regex that can be used to filter the files in the folder. This is passed to Pathlib.glob()
            legacy_support: This indicates how to deal with old firmware versions.
                If `error`: An error is raised, if an unsupported version is detected.
                If `warn`: A warning is raised, but the file is parsed without modification
                If `resolve`: A legacy conversion is performed to load old files. If no suitable conversion is found,
                    an error is raised. See the `legacy` package and the README to learn more about available
                    conversions.
        """
        return cls.from_file_paths(Path(base_path).glob(filter_pattern), legacy_support=legacy_support)

    def calibrate_imu(self: T, calibrations: Iterable[Union['CalibrationInfo', path_t]], inplace: bool = False) -> T:
        """Calibrate the imus of all datasets by providing a list of calibration infos.

        Args:
            calibrations: List of calibration infos in the same order than `self.datasets`
            inplace: If True this methods modifies the current session object. If False, a copy of the sesion and all
                dataset objects is created

        See Also:
            :py:meth:`NilsPodLib.dataset.Dataset.calibrate_imu`

        """
        s = inplace_or_copy(self, inplace)
        s.datasets = [d.calibrate_imu(c, inplace=True) for d, c in zip(s.datasets, calibrations)]
        return s

    def calibrate_acc(self: T, calibrations: Iterable[Union['CalibrationInfo', path_t]], inplace: bool = False) -> T:
        """Calibrate the accs of all datasets by providing a list of calibration infos.

        Args:
            calibrations: List of calibration infos in the same order than `self.datasets`
            inplace: If True this methods modifies the current session object. If False, a copy of the sesion and all
                dataset objects is created

        See Also:
            :py:meth:`NilsPodLib.dataset.Dataset.calibrate_acc`

        """
        s = inplace_or_copy(self, inplace)
        s.datasets = [d.calibrate_acc(c, inplace=True) for d, c in zip(s.datasets, calibrations)]
        return s

    def calibrate_gyro(self: T, calibrations: Iterable[Union['CalibrationInfo', path_t]], inplace: bool = False) -> T:
        """Calibrate the gyros of all datasets by providing a list of calibration infos.

        Args:
            calibrations: List of calibration infos in the same order than `self.datasets`
            inplace: If True this methods modifies the current session object. If False, a copy of the sesion and all
                dataset objects is created

        See Also:
            :py:meth:`NilsPodLib.dataset.Dataset.calibrate_acc`

        """
        s = inplace_or_copy(self, inplace)
        s.datasets = [d.calibrate_gyro(c, inplace=True) for d, c in zip(s.datasets, calibrations)]
        return s

    def _cascading_dataset_method_called(self, name: str, *args, **kwargs):
        return_vals = tuple(getattr(d, name)(*args, **kwargs) for d in self.datasets)
        if all(isinstance(d, Dataset) for d in return_vals):
            inplace = kwargs.get('inplace', False)
            s = inplace_or_copy(self, inplace)
            s.datasets = return_vals
            return s
        return return_vals

    def _cascading_dataset_attribute_access(self, name: str) -> Any:
        return_val = tuple([getattr(d, name) for d in self.datasets])
        if name == 'info':
            return ProxyHeader(return_val)
        return return_val


class SyncedSession(Session):
    VALIDATE_ON_INIT = True
    _fully_synced = False

    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)
        if self.VALIDATE_ON_INIT:
            self.validate()

    def validate(self) -> None:
        """Check if basic properties of a synced session are fulfilled.

        Raises:
            ValueError: This raises a ValueError in the following cases:
                - One or more of the datasets are not part of the same syncgroup/same sync channel
                - Multiple datasets are marked as "master"
                - One or more datasets indicate that they are not synchronised
                - One or more dataset has a different sampling rate than the others
                - If the recording times of provided datasets do not have any overlap

        """
        if not self._validate_sync_groups():
            raise ValueError('The provided datasets are not part of the same sync_group')
        master_valid, slaves_valid = self._validate_sync_role()
        if not master_valid:
            raise ValueError('SyncedSessions require exactly 1 master.')
        if not slaves_valid:
            raise ValueError('One of the provided sessions is not correctly set to either slave or master')
        if not self._validate_sampling_rate():
            raise ValueError('All provided sessions need to have the same sampling rate')
        if not self._validate_overlapping_record_time():
            raise ValueError('The provided datasets do not have any overlapping time period.')

    def _validate_sync_groups(self) -> bool:
        """Check that all _headers belong to the same sync group."""
        sync_channel = set(self.info.sync_channel)
        sync_address = set(self.info.sync_address)
        return all((len(i) == 1 for i in [sync_channel, sync_address]))

    def _validate_sync_role(self) -> Tuple[bool, bool]:
        """Check that there is only 1 master and all other sensors were configured as slaves."""
        roles = self.info.sync_role
        master_valid = len([i for i in roles if i == 'master']) == 1
        slaves_valid = len([i for i in roles if i == 'slave']) == len(roles) - 1
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

    @property
    def master(self) -> Dataset:
        """Return the master dataset of the session."""
        return next(d for d in self.datasets if d.info.sync_role == 'master')

    @property
    def slaves(self) -> Tuple[Dataset]:
        """Return a list of all slave datasets in the session."""
        return tuple(d for d in self.datasets if d.info.sync_role == 'slave')

    def cut_to_syncregion(self: Type[T], end: bool = False, only_to_master: bool = False,
                          warn_thres: Optional[int] = 30, inplace: bool = False) -> T:
        """Cut all datasets to the regions where they were synchronised to the master.

        Args:
            only_to_master: If True each slave will be cut to the region, where it was synchronised with the master.
                Master will not be changed. If False, all sensors will be cut to the region, where ALL sensors where
                in sync. Only in the latter case all datasets will have the same length and are guarantied to have the
                same counter.
            end: Whether the dataset should be cut at the `info.last_sync_index`. Usually it can be assumed that the
                data will be synchronous for multiple seconds after the last sync package. Therefore, it might be
                acceptable to just ignore the last syncpackage and just cut the start of the dataset.
            warn_thres: Threshold in seconds from the end of a dataset. If the last syncpackage occurred more than
                warn_thres before the end of the dataset, a warning is emitted. Use warn_thres = None to silence.
                This is not relevant if the end of the dataset is cut (e.g. `end=True`)
            inplace: If operation should be performed on the current Session object, or on a copy

        Warns:
            If a syncpackage occurred far before the last sample in any of the dataset. See arg `warn_thres`.
        """
        s = inplace_or_copy(self, inplace)
        if warn_thres is not None and end is not True:
            sync_warn = [d.info.sensor_id for d in s.slaves if d._check_sync_packages(warn_thres) is False]
            if any(sync_warn):
                warnings.warn('For the sensors with the ids {} the last syncpackage occurred more than {} s before the '
                              'end of the dataset. The last section of this data should not be trusted.'.format(
                    sync_warn, warn_thres))

        s = super(SyncedSession, s).cut_to_syncregion(end=end, inplace=True, warn_thres=None)
        if only_to_master is True:
            return s

        start_idx = [d.counter[0] for d in s.datasets]
        stop_idx = [d.counter[-1] for d in s.datasets]
        if not validate_existing_overlap(np.array(start_idx), np.array(stop_idx)):
            raise ValueError('The provided datasets do not have a overlapping regions where all are synced!')

        s = super(SyncedSession, s).cut_counter_val(np.max(start_idx), inplace=True)
        stop_idx = [len(d.counter) for d in s.datasets]
        s = super(SyncedSession, s).cut(stop=np.min(stop_idx), inplace=True)

        # Finally set the master counter to all slaves to really ensure identical counters
        for d in s.slaves:
            d.counter = s.master.counter
        self._fully_synced = True
        return s

    def data_as_df(self, datastreams: Optional[Sequence[str]] = None, index: Optional[str] = None,
                   concat_df: Optional[bool] = False):
        """Export all datasets of the dataset in a list of (or a single) pandas DataFrame.

        See Also:
            :py:meth:`NilsPodLib.dataset.Dataset.data_as_df`

        Args:
            datastreams: Optional list of datastream names, if only specific ones should be included. Datastreams that
                are not part of the current dataset will be silently ignored.
            index: Specify which index should be used for the dataset. The options are:
                "counter": For the actual counter
                "time": For the time in seconds since the first sample
                "utc": For the utc time stamp of each sample
                "utc_datetime": for a pandas DateTime index in UTC time
                None: For a simple index (0...N)
            concat_df: If True the individual dfs from each dataset will be concatenated. This is only supported, if the
                session is properly cut to the sync region and all the datasets have the same counter.

        Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Raises:
            ValueError: If any other than the allowed `index` values are used.
        """
        import pandas as pd
        dfs = super().data_as_df(datastreams, index)
        if concat_df is True:
            if not self._fully_synced:
                raise SynchronisationError('Only fully synced datasets, can be exported as a df with unified index.')
            dfs = pd.concat(dfs, axis=1, keys=self.info.sensor_id)
        return dfs

    def imu_data_as_df(self, index: Optional[str] = None, concat_df: Optional[bool] = False):
        """Export the acc and gyro datastreams of all datasets in list of (or a single) pandas DataFrame.

        See Also:
            :py:meth:`NilsPodLib.session.SyncedSession.data_as_df`
            :py:meth:`NilsPodLib.dataset.Dataset.data_as_df`
            :py:meth:`NilsPodLib.dataset.Dataset.imu_data_as_df`


        Args:
            index: Specify which index should be used for the dataset. The options are:
                "counter": For the actual counter
                "time": For the time in seconds since the first sample
                "utc": For the utc time stamp of each sample
                "utc_datetime": for a pandas DateTime index in UTC time
                None: For a simple index (0...N)
            concat_df: If True the individual dfs from each dataset will be concatenated. This is only supported, if the
                session is properly cut to the sync region and all the datasets have the same counter.

        Notes:
            This method calls the `data_as_df` methods of each Datastream object and then concats the results.
            Therefore, it will use the column information of each datastream.

        Raises:
            ValueError: If any other than the allowed `index` values are used.

        """
        return self.data_as_df(datastreams=['acc', 'gyro'], index=index)
