# -*- coding: utf-8 -*-
"""Session groups multiple Datasets from sensors recorded at the same time
Created on Thu Sep 28 11:32:22 2017

@author: Nils Roth, Arne Küderle
"""

import copy
from typing import Iterable, Tuple

import numpy as np

from NilsPodLib.dataset import Dataset, ProxyDataset
from NilsPodLib.header import Header, ProxyHeader


# TODO: Concept of inplace for sessions
# TODO: Calibration for multiple sensors
# TODO: Helper to create from folder/multiple names
from NilsPodLib.utils import validate_existing_overlap, inplace_or_copy


class Session:
    datasets: ProxyDataset

    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = ProxyDataset(tuple(datasets))

    @property
    def info(self) -> ProxyHeader:
        return ProxyHeader(headers=tuple(self.datasets.info))

    def calibrate(self):
        self.leftFoot.calibrate()
        self.rightFoot.calibrate()

    @classmethod
    def from_filePaths(cls, leftFootPath, rightFootPath):
        leftFoot = Dataset(leftFootPath, Header, freeRTOS)
        rightFoot = Dataset(rightFootPath, Header, freeRTOS)
        session = cls(leftFoot, rightFoot)
        return session

    @classmethod
    def from_folderPath(cls, folderPath):
        [leftFootPath, rightFootPath] = getFilesNamesPerFoot(folderPath)
        leftFoot = Dataset(leftFootPath)
        rightFoot = Dataset(rightFootPath)
        session = cls(leftFoot, rightFoot)
        return session


class SyncedSession(Session):

    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)
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
        """Check that all _headers belong to the same sync group"""
        sync_group = set(self.info.sync_group)
        sync_channel = set(self.info.sync_channel)
        sync_address = set(self.info.sync_address)
        return all((True for i in [sync_group, sync_channel, sync_address] if len(i) == 1))

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
        return next(d for d in self.datasets if d.info.sync_role == 'master')

    @property
    def slaves(self) -> Tuple[Dataset]:
        return tuple(d for d in self.datasets if d.info.sync_role == 'slaves')

    def synchronize(self, only_to_master: bool = False, inplace=False):
        """Cut all datasets to the regions where they were syncronised to the master.

        Args:
            only_to_master: If True each slave will be cut to the region, where it was synchronised with the master.
                Master will not be changed. If False, all sensors will be cut to the region, where ALL sensors where
                in sync
        """
        # TODO: Replace all counter arrays with the master counter (is this required?)
        # cut all individual sensors
        s = inplace_or_copy(self, inplace)

        if only_to_master is True:
            s.datasets = s.datasets.cut_to_syncregion()
            return s

        start_idx = [d.info.sync_index_start for d in s.slaves]
        stop_idx = [d.info.sync_index_stop for d in s.slaves]
        if not validate_existing_overlap(np.array(start_idx), np.array(stop_idx)):
            raise ValueError('The provided datasets do not have a overlapping regions where all a synced!')

        s.datasets = s.datasets.cut(np.max(start_idx), np.min(stop_idx))
        return s
