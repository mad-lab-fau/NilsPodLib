#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import warnings

import numpy as np

from NilsPodLib.utils import convert_little_endian


# TODO: Put all Metainfos about the sensors into one object
# TODO: Add method that can output all the header info as json
# TODO: Include metainformation for units of sensors
class Header:
    """Additional Infos of recording.

    Note:
        - utc timestamps and datetime, might not be in UTC. We just provide the values recorded by the sensor without
            any local conversions
    """
    enabled_sensors: tuple

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

    unix_time_start: int
    unix_time_stop: int

    version_firmware: str
    mac_address: str

    custom_meta_data = np.zeros(4)
    # Note: the number of samples might not be equal to the actual number of samples in the file, because the sensor
    #   only transmits full flash pages. This means a couple of samples (max. 2048/sample_size) at the end might be cut.
    num_samples: int

    _SENSOR_FLAGS = {
        'acc': 0x01,
        'gyro': 0x02,
        'mag': 0x04,
        'baro': 0x08,
        'analog': 0x10,
        'ecg': 0x20,
        'ppg': 0x40,
        'battery': 0x80
    }

    _SENSOR_SAMPLE_LENGTH = {
        'acc': (6, 3),
        'gyro': (6, 3),
        'mag': (6, 3),
        'baro': (2, 1),
        'analog': (3, 1),
        'ecg': (None, None),  # Needs to be implement
        'ppg': (None, None),  # Needs to be implement
        'battery': (1, 1)

    }

    _SENSOR_LEGENDS = {
        'acc': tuple('acc_' + x for x in 'xyz'),
        'gyro': tuple('gyr_' + x for x in 'xyz')
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

    _header_fields = ['enabled_sensors', 'motion_interrupt_enabled', 'dock_mode_enabled', 'sensor_position',
                      'session_termination', 'sample_size', 'sampling_rate_hz', 'acc_range_g', 'gyro_range_dps',
                      'sync_role', 'sync_distance_ms', 'sync_group', 'sync_address', 'sync_channel', 'sync_index_start',
                      'sync_index_stop', 'unix_time_start', 'unix_time_stop', 'version_firmware', 'mac_address',
                      'custom_meta_data', 'num_samples']

    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            if k in self._header_fields:
                setattr(self, k, v)
            else:
                # Should this be a error?
                warnings.warn('Unexpected Argument {} for Header'.format(k))

    @classmethod
    def from_bin_array(cls, bin_array: np.ndarray):
        header_dict = cls.parse_header_package(bin_array)
        return cls(**header_dict)

    @classmethod
    def parse_header_package(cls, bin_array: np.ndarray):
        # Note that because the info packet already has the first byte (info size) removed, all byte numbers are
        # shifted compared to the documentation
        header_dict = dict()

        header_dict['sample_size'] = bin_array[0]

        sensors = bin_array[1]
        enabled_sensors = list()
        for para, val in cls._SENSOR_FLAGS.items():
            if bool(sensors & val) is True:
                enabled_sensors.append(para)
        header_dict['enabled_sensors'] = tuple(enabled_sensors)

        header_dict['sampling_rate_hz'] = cls._SAMPLING_RATES[bin_array[2] & 0x0F]

        header_dict['session_termination'] = next(
            k for k, v in cls._SESSION_TERMINATION.items() if bool(bin_array[3] & v) is True)

        header_dict['sync_role'] = cls._SYNC_ROLE[bin_array[4]]

        header_dict['sync_distance_ms'] = bin_array[5] * 100.0

        header_dict['sync_group'] = bin_array[6]

        header_dict['acc_range_g'] = bin_array[7]

        header_dict['gyro_range_dps'] = bin_array[8] * 125

        sensor_position = bin_array[9]
        header_dict['sensor_position'] = cls._SENSOR_POS.get(sensor_position, sensor_position)

        operation_mode = bin_array[10]
        for para, val in cls._OPERATION_MODES.items():
            header_dict[para] = bool(operation_mode & val)

        header_dict['custom_meta_data'] = bin_array[11:14]

        # Note: We ignore timezones and provide just the time info, which was stored in the sensor
        header_dict['unix_time_start'] = convert_little_endian(bin_array[14:18])
        header_dict['unix_time_stop'] = convert_little_endian(bin_array[18:22])

        header_dict['num_samples'] = convert_little_endian(bin_array[22:26])

        header_dict['sync_index_start'] = convert_little_endian(bin_array[26:30])
        header_dict['sync_index_stop'] = convert_little_endian(bin_array[30:34])

        header_dict['mac_address'] = ':'.join([hex(int(x))[-2:] for x in bin_array[34:40]][::-1])

        header_dict['sync_address'] = ''.join([hex(int(x))[-2:] for x in bin_array[40:45]][::-1])
        header_dict['sync_channel'] = bin_array[45]

        header_dict['version_firmware'] = 'v{}.{}.{}'.format(*(int(x) for x in bin_array[-3:]))

        return header_dict

    @property
    def duration_s(self) -> int:
        return self.unix_time_stop - self.unix_time_start

    @property
    def utc_datetime_start(self) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(self.unix_time_start)

    @property
    def utc_datetime_stop(self) -> datetime.datetime:
        return datetime.datetime.utcfromtimestamp(self.unix_time_stop)

    @property
    def datetime_start(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.unix_time_start)

    @property
    def datetime_stop(self) -> datetime.datetime:
        return datetime.datetime.fromtimestamp(self.unix_time_stop)

    @property
    def is_synchronised(self) -> bool:
        return not self.sync_role == 'disabled'

    @property
    def has_position_info(self) -> bool:
        return not self.sensor_position == 'undefined'

    @property
    def sensor_id(self) -> str:
        return ''.join(self.mac_address[-5:].split(':'))
