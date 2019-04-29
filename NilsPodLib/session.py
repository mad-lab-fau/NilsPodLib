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


# TODO: Session synced
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

    def _validate_sync_groups(self):
        """Check that all _headers belong to the same sync group"""
        sync_group = set(self.info.sync_group)
        sync_channel = set(self.info.sync_channel)
        sync_address = set(self.info.sync_address)
        return all((True for i in [sync_group, sync_channel, sync_address] if len(i) == 1))

    def _validate_sync_role(self):
        """Check that there is only 1 master and all other sensors were configured as slaves."""
        roles = self.info.sync_role
        master_valid = len([i for i in roles if i == 'master']) == 1
        slaves_valid = len([i for i in roles if i == 'slaves']) == len(roles) - 1
        return master_valid, slaves_valid

    def _validate_sampling_rate(self):
        """Check that all sensors had the same sampling rate."""
        sr = set(self.info.sampling_rate_hz)
        return len(sr) == 1

    @property
    def master(self) -> Dataset:
        return next(d for d in self.datasets if d.info.sync_role == 'master')

    @property
    def slaves(self) -> Tuple[Dataset]:
        return tuple(d for d in self.datasets if d.info.sync_role == 'slaves')

    def synchronize(self):
        try:
            if self.leftFoot.header.syncRole == self.rightFoot.header.syncRole:
                print("ERROR: no master/slave pair found - synchronization FAILED")
                return

            if self.rightFoot.header.syncRole == 'master':
                master = self.rightFoot
                slave = self.leftFoot
            else:
                master = self.leftFoot
                slave = self.rightFoot

            try:
                inSync = (np.argwhere(slave.sync > 0)[0])[0]
            except:
                print("No Synchronization signal found - synchronization FAILED")
                return

            # cut away all sample at the beginning until both data streams are synchronized (SLAVE)
            inSync = (np.argwhere(slave.sync > 0)[0])[0]
            slave = slave.cut_dataset(inSync, len(slave.counter))

            # cut away all sample at the beginning until both data streams are synchronized (MASTER)
            inSync = (np.argwhere(master.counter >= slave.counter[0])[0])[0]
            master = master.cut_dataset(inSync, len(master.counter))

            # cut both streams to the same lenght
            if len(master.counter) >= len(slave.counter):
                length = len(slave.counter) - 1
            else:
                length = len(master.counter) - 1

            slave = slave.cut_dataset(0, length)
            master = master.cut_dataset(0, length)

            if self.rightFoot.header.syncRole == 'master':
                self.rightFoot = master
                self.leftFoot = slave
            else:
                self.rightFoot = slave
                self.leftFoot = master
            # check if synchronization is valid
            # test synchronization
            deltaCounter = abs(self.leftFoot.counter - self.rightFoot.counter)
            sumDelta = np.sum(deltaCounter)
            if sumDelta != 0.0:
                print("ATTENTION: Error in synchronization. Check Data!")
        except Exception as e:
            print(e)
            print("synchronization failed with ERROR")
