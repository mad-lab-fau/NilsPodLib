#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import numpy as np


class Header:
    # default Session Header
    gyro_enabled: bool
    acc_enabled: bool = True
    pressure_enabled: bool
    baro_enabled: bool
    battery_enabled: bool

    sampling_rate_hz: float
    acc_range_g: float
    gyro_range_dps: float

    sync_role = 'disabled'
    sync_distance_ms: float

    sensor_position = 'undefined'

    datetime_start: datetime.datetime
    datetime_stop: datetime.datetime
    unix_time_start = 0
    unix_time_stop = 0

    version_firmware: str
    packet_size: int
    low_voltage_termination: bool

    meta_data = np.zeros(4)
    num_samples: int

    _SENSOR_FLAGS = {
        'gyro_enabled': 0x01,
        'pressure_enabled': 0x02,
        'baro_enabled': 0x04,
        'battery_enabled': 0x08
    }

    _SAMPLING_RATES = {
        10: 102.4,
        5: 204.8,
        4: 256.0,
        2: 512.0,
        1: 1024.0
    }

    _SYNC_ROLE = {
        1: 'slave',
        2: 'master'
    }

    _SENSOR_POS = {
        1: 'left foot',
        2: 'right foot',
        3: 'hip'
    }

    def __init__(self, header_packet=None):
        if header_packet is not None:
            self.packet_size = header_packet[0]
            sensors = header_packet[1]

            for para, val in self._SENSOR_FLAGS.items():
                setattr(self, para, bool(sensors & val))

            self.sampling_rate_hz = self._SAMPLING_RATES[header_packet[2] & 0x0F]
            self.samplingTime_ms = (1.0 / self.sampling_rate_hz) * 1000.0

            self.low_voltage_termination = bool(header_packet[2] & 0x80)

            self.sync_role = self._SYNC_ROLE.get(header_packet[3], self.sync_role)

            self.sync_distance_ms = header_packet[4] * 100.0

            self.acc_range_g = header_packet[6]

            self.gyro_range_dps = header_packet[7] * 125

            self.sensor_position = self._SENSOR_POS.get(header_packet[8], self.sensor_position)

            self.meta_data = header_packet[9:13]

            packed_date_time = int(header_packet[13]) | (int(header_packet[14]) << 8) | (
                        int(header_packet[15]) << 16) | (
                                       int(header_packet[16]) << 24)
            self.unix_time_start = packed_date_time
            self.datetime_start = datetime.datetime.fromtimestamp(self.unix_time_start)

            packed_date_time = int(header_packet[17]) | (int(header_packet[18]) << 8) | (
                        int(header_packet[19]) << 16) | (
                                       int(header_packet[20]) << 24)
            self.unix_time_stop = packed_date_time
            self.datetime_stop = datetime.datetime.fromtimestamp(self.unix_time_stop)

            self.num_samples = int(header_packet[21]) | (int(header_packet[22]) << 8) | (
                        int(header_packet[23]) << 16) | (
                                      int(header_packet[24]) << 24)

            self.version_firmware = "V" + str(int(header_packet[-3])) + "." + str(int(header_packet[-2])) + "." + str(
                int(header_packet[-1]))
