# -*- coding: utf-8 -*-
"""Session groups multiple Datasets from sensors recorded at the same time
Created on Thu Sep 28 11:32:22 2017

@author: Nils Roth, Arne Küderle
"""

import copy
from typing import Iterable

import numpy as np
import pandas as pd

from NilsPodLib.dataset import Dataset, ProxyDataset
from NilsPodLib.header import Header, ProxyHeader


# TODO: Session synced
# TODO: Synced session as separate class?
class Session:
    datasets: ProxyDataset

    def __init__(self, datasets: Iterable[Dataset]):
        self.datasets = ProxyDataset(datasets)

    @property
    def info(self) -> ProxyHeader:
        return ProxyHeader(headers=self.datasets.info)

    def calibrate(self):
        self.leftFoot.calibrate()
        self.rightFoot.calibrate()


class SyncedSession(Session):

    def __init__(self, datasets: Iterable[Dataset]):
        super().__init__(datasets)
        if not self._validate_sync_groups():
            raise ValueError('The providid _headers are not part of the same sync_group')
        master_valid, slaves_valid = self._validate_sync_role()
        if not master_valid:
            raise ValueError('SyncedSessions require exactly 1 master.')
        if not slaves_valid:
            raise ValueError('One of the provided sessions is not correctly set to either slave or master')

    def _validate_sync_groups(self):
        """Check that all _headers belong to the same sync group"""
        sync_group = {d.info.sync_group for d in self.datasets}
        sync_channel = {d.info.sync_channel for d in self.datasets}
        sync_address = {d.info.sync_address for d in self.datasets}
        return all((True for i in [sync_group, sync_channel, sync_address] if len(i) == 1))

    def _validate_sync_role(self):
        """Check that there is only 1 master and all other sensors were configured as slaves."""
        roles = [d.info.sync_role for d in self.datasets]
        master_valid = len([i for i in roles if i == 'master']) == 1
        slaves_valid = len([i for i in roles if i == 'slaves']) == len(roles) - 1
        return master_valid, slaves_valid

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

    def synchronizeFallback(self):
        # cut away all sample at the beginning until both data streams are synchronized (SLAVE)
        inSync = (np.argwhere(self.leftFoot.sync > 0)[0])[0]
        self.leftFoot = self.leftFoot.cut_dataset(inSync, len(self.leftFoot.counter))

        # cut away all sample at the beginning until both data streams are synchronized (MASTER)
        inSync = (np.argwhere(self.rightFoot.counter >= self.leftFoot.counter[0])[0])[0]
        self.rightFoot = self.rightFoot.cut_dataset(inSync, len(self.rightFoot.counter))

        # cut both streams to the same lenght
        if len(self.rightFoot.counter) >= len(self.leftFoot.counter):
            length = len(self.leftFoot.counter) - 1
        else:
            length = len(self.rightFoot.counter) - 1

        self.leftFoot = self.leftFoot.cut_dataset(0, length)
        self.rightFoot = self.rightFoot.cut_dataset(0, length)

    def synchronize(self):
        if self.leftFoot.header.syncRole == 'disabled' or self.rightFoot.header.syncRole == 'disabled':
            print("No Header information found using fallback sync")
            self.synchronizeFallback()
            return
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

    def rotateAxis(self, system):
        if system == 'egait':
            self.leftFoot.rotate_axis('gyro', 2, 0, 1, -1, -1, 1)  # swap axis Z,X,Y, change sign -X-Y+Z
            self.leftFoot.rotate_axis('acc', 2, 0, 1, -1, -1, 1)
            self.rightFoot.rotate_axis('gyro', 2, 0, 1, 1, 1, -1)
            self.rightFoot.rotate_axis('acc', 2, 0, 1, 1, -1, -1)
            self.leftFoot.rotate_axis('pressure', 0, 0, 0, 0, 0, 0)
            self.rightFoot.rotate_axis('pressure', 0, 0, 0, 0, 0, 0)
        elif system == 'default':
            self.rightFoot.rotate_axis('default', 0, 0, 0, 0, 0, 0)
            self.leftFoot.rotate_axis('default', 0, 0, 0, 0, 0, 0)
        else:
            print('unknown system, you need to handle axis rotation per foot yourself!')

    def cutData(self, start, stop):
        session = copy.copy(self)
        session.leftFoot = session.leftFoot.cut_dataset(start, stop)
        session.rightFoot = session.rightFoot.cut_dataset(start, stop)
        return session

    def convertToDataFrame(self, session):
        # create pandas dataframe
        dataset = session.leftFoot
        baro = np.reshape(dataset.baro.data, (dataset.size, 1))
        battery = np.reshape(dataset.battery.data, (dataset.size, 1))
        rtc = np.reshape(dataset.rtc, (dataset.size, 1))
        dfLeft = pd.DataFrame(
            np.hstack((rtc, dataset.acc.data, dataset.gyro.data, dataset.pressure.data, baro, battery)))
        foot = 'L'
        dfLeft.columns = ['utc' + foot, 'aX' + foot, 'aY' + foot, 'aZ' + foot, 'gX' + foot, 'gY' + foot, 'gZ' + foot,
                          'p1' + foot, 'p2' + foot, 'p3' + foot, 'b' + foot, 'bat' + foot]

        dataset = session.rightFoot
        baro = np.reshape(dataset.baro.data, (dataset.size, 1))
        battery = np.reshape(dataset.battery.data, (dataset.size, 1))
        rtc = np.reshape(dataset.rtc, (dataset.size, 1))
        dfRight = pd.DataFrame(
            np.hstack((rtc, dataset.acc.data, dataset.gyro.data, dataset.pressure.data, baro, battery)))
        foot = 'R'
        dfRight.columns = ['utc' + foot, 'aX' + foot, 'aY' + foot, 'aZ' + foot, 'gX' + foot, 'gY' + foot, 'gZ' + foot,
                           'p1' + foot, 'p2' + foot, 'p3' + foot, 'b' + foot, 'bat' + foot]
        dfCombined = pd.concat([dfLeft, dfRight], axis=1)

        return dfCombined

    def saveToCSV(self, filename, sep=';', compression=None, index=False):
        df = self.convertToDataFrame(self)
        df.to_csv(filename, sep=sep, compression=compression)
