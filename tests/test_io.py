import datetime
from pathlib import Path
import numpy as np

from NilsPodLib.datastream import Datastream
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_DATA = HERE / 'test_data'


def test_load_simple():
    path = TEST_DATA / 'simple_synced_master.bin'
    dataset = Dataset.from_bin_file(path=path)

    # Toplevel Stuff
    assert dataset.path == path
    assert isinstance(dataset.acc, Datastream)
    assert dataset.acc.is_calibrated is False
    assert dataset.gyro is None
    assert dataset.baro is None
    assert dataset.mag is None
    assert dataset.battery is None
    assert dataset.analog is None
    assert dataset.ppg is None
    assert dataset.ecg is None
    assert dataset.ACTIVE_SENSORS == ('acc',)

    # Header
    info = dataset.info
    assert info.sample_size == 10
    assert info.enabled_sensors == ('acc',)
    assert info.sampling_rate_hz == 204.8
    assert info.session_termination == 'BLE'
    assert info.acc_range_g == 16
    assert info.gyro_range_dps == 2000
    assert info.unix_time_start == 1556025376
    assert info.utc_datetime_start == datetime.datetime(2019, 4, 23, 13, 16, 16)
    assert info.datetime_start == datetime.datetime(2019, 4, 23, 15, 16, 16)
    assert info.unix_time_stop == 1556025420
    assert info.utc_datetime_stop == datetime.datetime(2019, 4, 23, 13, 17, 00)
    assert info.datetime_stop == datetime.datetime(2019, 4, 23, 15, 17, 00)
    assert info.duration_s == 44
    assert info.num_samples == 9146
    assert info.version_firmware == 'v0.12.0'
    assert info.sync_role == 'master'
    assert info.is_synchronised is True
    assert info.sync_group == 9
    assert info.sync_index_start == 0
    assert info.sync_index_stop == 0
    assert info.mac_address == '3a:x9:23:21:9e:82'
    assert info.sensor_id == '9e82'
    assert info.sync_address == '9f2be06f7c'
    assert info.sync_channel == 43

    # System Info
    assert info.sensor_position == 'undefined'
    assert info.has_position_info is False
    assert info.dock_mode_enabled is False
    assert info.motion_interrupt_enabled is False
    assert np.array_equal(info.custom_meta_data, np.zeros(3))


def test_sync_info():
    path = TEST_DATA / 'simple_synced_slave.bin'
    dataset = Dataset.from_bin_file(path=path)
    info = dataset.info

    assert info.sync_role == 'slave'
    assert info.is_synchronised is True
    assert info.sync_group == 9  # Should be same as master (see test above)
    assert info.sync_index_start == 489
    assert info.sync_index_stop == 6636
    assert info.sync_address == '9f2be06f7c'  # Should be same as master (see test above)
    assert info.sync_channel == 43  # Should be same as master (see test above)

