#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import numpy as np

from NilsPodLib.utils import convert_little_endian


class Header:
    acc_enabled: bool
    gyro_enabled: bool
    magnetometer_enabled: bool
    baro_enabled: bool
    analog_enabled: bool
    ecg_enabled: bool
    ppg_enabled: bool
    battery_enabled: bool

    motion_interrupt_enabled: bool
    dock_mode_enabled: bool
    sensor_position: str
    session_termination: int
    sample_size: int

    sampling_rate_hz: float
    acc_range_g: float
    gyro_range_dps: float

    sync_role: str
    sync_distance_ms: float
    sync_group: int
    sync_address: int
    sync_channel: int
    sync_index_start: int
    sync_index_stop: int

    datetime_start: datetime.datetime
    datetime_stop: datetime.datetime
    unix_time_start: int
    unix_time_stop: int

    version_firmware: str
    mac_address: str

    custom_meta_data = np.zeros(4)
    num_samples: int

    _SENSOR_FLAGS = {
        'acc_enabled': 0x01,
        'gyro_enabled': 0x02,
        'magnetometer_enabled': 0x04,
        'baro_enabled': 0x08,
        'analog_enabled': 0x10,
        'ecg_enabled': 0x20,
        'ppg_enabled': 0x40,
        'battery_enabled': 0x80
    }

    _OPERATION_MODES = {
        'motion_interrupt_enabled': 0x80,
        'dock_mode_enabled': 0x40,
    }

    _SAMPLING_RATES = {
        10: 102.4,
        5: 204.8,
        4: 256.0,
        2: 512.0,
        1: 1024.0
    }

    _SESSION_TERMINATION = {
        'no memory': 0x10,
        'BLE': 0x20,
        'dock': 0x40,
        'low battery': 0x80
    }

    _SYNC_ROLE = {
        0: 'disabled',
        1: 'slave',
        2: 'master'
    }

    _SENSOR_POS = {
        0: 'undefined',
        1: 'left foot',
        2: 'right foot',
        3: 'hip',
        4: 'left wrist',
        5: 'right wrist',
        6: 'chest'
    }

    def __init__(self, header_packet=None):

        # Note that because the info packet already has the first byte (info size) removed, all byte numbers are
        # shifted compared to the documentation
        if header_packet is not None:
            self.sample_size = header_packet[0]

            sensors = header_packet[1]
            for para, val in self._SENSOR_FLAGS.items():
                setattr(self, para, bool(sensors & val))

            self.sampling_rate_hz = self._SAMPLING_RATES[header_packet[2] & 0x0F]
            self.samplingTime_ms = (1.0 / self.sampling_rate_hz) * 1000.0

            self.session_termination = next(
                k for k, v in self._SESSION_TERMINATION.items() if bool(header_packet[3] & v) is True)

            self.sync_role = self._SYNC_ROLE[header_packet[4]]

            self.sync_distance_ms = header_packet[5] * 100.0

            self.sync_group = header_packet[6]

            self.acc_range_g = header_packet[7]

            self.gyro_range_dps = header_packet[8] * 125

            # self.sensor_position = self._SENSOR_POS.get(header_packet[8], self.sensor_position)

            sensor_position = header_packet[9]
            self.sensor_position = self._SENSOR_POS.get(sensor_position, sensor_position)

            operation_mode = header_packet[10]
            for para, val in self._OPERATION_MODES.items():
                setattr(self, para, bool(operation_mode & val))

            self.custom_meta_data = header_packet[11:14]

            self.unix_time_start = convert_little_endian(header_packet[14:18])
            self.datetime_start = datetime.datetime.fromtimestamp(self.unix_time_start)

            self.unix_time_stop = convert_little_endian(header_packet[18:22])
            self.datetime_stop = datetime.datetime.fromtimestamp(self.unix_time_stop)

            self.num_samples = convert_little_endian(header_packet[22:26])

            self.sync_index_start = convert_little_endian(header_packet[26:30])
            self.sync_index_stop = convert_little_endian(header_packet[30:34])

            self.mac_address = ':'.join([hex(int(x))[-2:] for x in header_packet[34:40]][::-1])

            self.sync_address = ''.join([hex(int(x))[-2:] for x in header_packet[40:45]][::-1])
            self.sync_channel = header_packet[45]

            self.version_firmware = 'v{}.{}.{}'.format(*(int(x) for x in header_packet[-3:]))

    @property
    def duration_s(self) -> int:
        return self.unix_time_stop - self.unix_time_start

    @property
    def is_synchronised(self) -> bool:
        return not self.sync_role == 'disabled'

    @property
    def has_position_info(self) -> bool:
        return not self.sensor_position == 'undefined'

    @property
    def sensor_id(self) -> str:
        return ''.join(self.mac_address[-5:].split(':'))

