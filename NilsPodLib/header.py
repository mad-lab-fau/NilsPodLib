# -*- coding: utf-8 -*-
"""Header class(es), which are used to read and access basic information from a recorded session."""

import datetime
import json
import warnings
from collections import OrderedDict
from distutils.version import StrictVersion
from itertools import chain
from typing import Tuple, Any, List, Dict, Union

import numpy as np

from NilsPodLib.consts import SENSOR_POS
from NilsPodLib.utils import convert_little_endian


# TODO: Fix type issues when using proxy header
# TODO: Add docstrings fro attributes
class HeaderFields:
    """Base class listing all the attributes of a session header."""

    enabled_sensors: tuple

    motion_interrupt_enabled: bool
    dock_mode_enabled: bool
    sensor_position: str
    session_termination: str
    sample_size: int

    sampling_rate_hz: float
    acc_range_g: float
    gyro_range_dps: float

    sync_role: str
    sync_distance_ms: float
    sync_address: int
    sync_channel: int
    sync_index_start: int
    sync_index_stop: int

    utc_start: int
    utc_stop: int

    version_firmware: str
    version_hardware: str
    mac_address: str

    custom_meta_data: tuple
    # Note: the number of samples might not be equal to the actual number of samples in the file, because the sensor
    #   only transmits full flash pages. This means a couple of samples (max. 2048/sample_size) at the end might be cut.
    n_samples: int

    # Note this must correspond to the order they appear in the datapackage when activated
    _SENSOR_FLAGS = OrderedDict([
        ('gyro', (0x02, 0x00)),
        ('acc', (0x01, 0x00)),
        ('mag', (0x04, 0x00)),
        ('baro', (0x08, 0x00)),
        ('analog', (0x10, 0x00)),
        ('ecg', (0x20, 0x00)),
        ('ppg', (0x40, 0x00)),
        ('temperature', (0x80, 0x00))
    ])

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

    @property
    def _header_fields(self) -> List[str]:
        """List all header fields.

        This is a little hacky and relies on that the header fields are the only attributes that are type annotated.
        """
        return list(HeaderFields.__annotations__.keys())

    @property
    def duration_s(self) -> int:
        """Length of the measurement."""
        return self.utc_stop - self.utc_start

    @property
    def utc_datetime_start(self) -> datetime.datetime:
        """Start time as utc datetime."""
        return datetime.datetime.utcfromtimestamp(self.utc_start)

    @property
    def utc_datetime_stop(self) -> datetime.datetime:
        """Stop time as utc datetime."""
        return datetime.datetime.utcfromtimestamp(self.utc_stop)

    @property
    def utc_datetime_start_day_midnight(self) -> datetime.datetime:
        """UTC timestamp marking midnight of the recording date.

        This is useful, as the sensor internal counter gets reset at midnight.
        I.e. utc_datetime_start_day_midnight + counter[0] * sampling_rate should be utc_datetime_start
        """
        return datetime.datetime.combine(self.utc_datetime_start.date(), datetime.time(), tzinfo=datetime.timezone.utc)

    @property
    def is_synchronised(self) -> bool:
        """If a recording was syncronised or not.

        Note:
            This does only indicate, that the session was recorded with the sync feature enabled, not that the data is
            actually synchronised.
        """
        return not self.sync_role == 'disabled'

    @property
    def has_position_info(self) -> bool:
        """If any information about the sensor position is provided."""
        return not self.sensor_position == 'undefined'

    @property
    def sensor_id(self) -> str:
        """Get the unique sensor identifier."""
        return ''.join(self.mac_address[-5:].split(':'))

    @property
    def strict_version_firmware(self) -> StrictVersion:
        """Get the firmware as a StrictVersion object."""
        return StrictVersion(self.version_firmware[1:])


class Header(HeaderFields):
    """Additional Infos of recording.

    Note:
        Usually their is no need to use this class on its own, but it is just used as a convenient wrapper to access
        all information via a dataset instance.

    Note:
        - utc timestamps and datetime, might not be in UTC. We just provide the values recorded by the sensor without
            any local conversions
    """

    def __init__(self, **kwargs):
        """Initialize a header object.

        This will just put all values provided in kwargs as attributes onto the class instance.
        If one value has an unexpected name, a warning is raised, and the key is ignored.
        """
        for k, v in kwargs.items():
            if k in self._header_fields:
                setattr(self, k, v)
            else:
                # Should this be a error?
                warnings.warn('Unexpected Argument {} for Header'.format(k))

    @classmethod
    def from_bin_array(cls, bin_array: np.ndarray) -> 'Header':
        """Create a new Header instance from an array of bytes."""
        header_dict = cls.parse_header_package(bin_array)
        return cls(**header_dict)

    @classmethod
    def from_json(cls, json_string: str) -> 'Header':
        """Create a new Header from a json export of the header.

        This is only tested with the direct output of the `to_json` method and should only be used to reimport a Header
        exported with this method.
        """
        h = cls(**json.loads(json_string))
        # ensure that the enabled sensors and custom_metadata have the right dtype
        for k in ('enabled_sensors', 'custom_meta_data'):
            if getattr(h, k, None):
                setattr(h, k, tuple(getattr(h, k)))
        return h

    def to_json(self) -> str:
        """Export a header as json.

        It can be imported again using the `from_json` method without information loss.
        """
        header_dict = {k: v for k, v in self.__dict__.items() if k in self._header_fields}
        return json.dumps(header_dict)

    @classmethod
    def parse_header_package(cls, bin_array: np.ndarray) -> Dict[str, Union[str, int, float, bool, tuple]]:
        """Extract all values from a binary header package."""
        # Note that because the info packet already has the first byte (info size) removed, all byte numbers are
        # shifted compared to the documentation
        bin_array = bin_array.astype(np.uint32)
        header_dict = dict()

        header_dict['sample_size'] = int(bin_array[0])

        sensors = bin_array[1:3]
        enabled_sensors = list()
        for para, val in cls._SENSOR_FLAGS.items():
            if bool(sensors[0] & val[0]) or bool(sensors[1] & val[1]):
                enabled_sensors.append(para)
        header_dict['enabled_sensors'] = tuple(enabled_sensors)

        # bin_array[2] = currently not used

        header_dict['sampling_rate_hz'] = cls._SAMPLING_RATES[bin_array[3] & 0x0F]

        header_dict['session_termination'] = next(
            k for k, v in cls._SESSION_TERMINATION.items() if bool(bin_array[4] & v) is True)

        header_dict['sync_role'] = cls._SYNC_ROLE[bin_array[5]]

        header_dict['sync_distance_ms'] = float(bin_array[6] * 100.0)

        header_dict['acc_range_g'] = float(bin_array[7])

        header_dict['gyro_range_dps'] = float(bin_array[8] * 125)

        sensor_position = bin_array[9]
        header_dict['sensor_position'] = SENSOR_POS.get(sensor_position, sensor_position)

        operation_mode = bin_array[10]
        for para, val in cls._OPERATION_MODES.items():
            header_dict[para] = bool(operation_mode & val)

        header_dict['custom_meta_data'] = tuple(bin_array[11:14].astype(float))

        # Note: We ignore timezones and provide just the time info, which was stored in the sensor
        header_dict['utc_start'] = int(convert_little_endian(bin_array[14:18]))
        header_dict['utc_stop'] = int(convert_little_endian(bin_array[18:22]))

        header_dict['n_samples'] = int(convert_little_endian(bin_array[22:26]))

        header_dict['sync_index_start'] = int(convert_little_endian(bin_array[26:30]))
        header_dict['sync_index_stop'] = int(convert_little_endian(bin_array[30:34]))

        header_dict['mac_address'] = ':'.join(['{:02x}'.format(int(x)) for x in bin_array[34:40]][::-1])

        header_dict['sync_address'] = ''.join(['{:02x}'.format(int(x)) for x in bin_array[40:45]][::-1])
        header_dict['sync_channel'] = int(bin_array[45])

        header_dict['version_hardware'] = ''.join((str(x) for x in bin_array[46:48]))

        header_dict['version_firmware'] = 'v{}.{}.{}'.format(*(int(x) for x in bin_array[-3:]))

        return header_dict


class _ProxyHeader(HeaderFields):
    """A proxy header used by session objects to get direct access to multiple headers.

    This allows to access attributes of multiple header instances without reimplementing all of its attributes.
    This is achieved by basically intercepting all getattribute calls and redirecting them to all header instances.

    This concept only allows read only access. However, usually their is no need to modify a header after creation.
    """

    _headers: Tuple[Header]

    def __init__(self, headers: Tuple[Header]):
        self._headers = headers

    def __getattribute__(self, name: str) -> Tuple[Any]:
        if name == '_headers':
            return super().__getattribute__(name)
        if callable(getattr(self._headers[0], name)) is True:
            raise ValueError(
                '_ProxyHeader only allows access to attributes of the info objects. {} is a callable method.'.format(
                    name))

        return tuple([getattr(d, name) for d in self._headers])

    def __setattr__(self, name: str, value: Any) -> None:
        if name == '_headers':
            return super().__setattr__(name, value)
        raise NotImplementedError('_ProxyHeader only allows readonly access to the info objects of a dataset')

    def __dir__(self):
        return chain(super().__dir__(), self._headers[0].__dir__())
