import datetime
from pathlib import Path
import numpy as np

from NilsPodLib.datastream import Datastream
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_DATA = HERE / 'test_data'


def test_load_simple():
    path = TEST_DATA / 'simple_test_data.bin'
    dataset = Dataset(path=path)

    # Toplevel Stuff
    assert dataset.path == path
    assert dataset.imu_is_calibrated is False
    assert isinstance(dataset.acc, Datastream)
    assert isinstance(dataset.gyro, Datastream)
    assert dataset.baro is None
    assert dataset.battery is None
    assert dataset.analog is None
    assert dataset.ACTIVE_SENSORS == ('acc', 'gyro')

    # Header
    info = dataset.info
    assert info.sample_size == 16
    assert info.acc_enabled is True
    assert info.gyro_enabled is True
    assert info.baro_enabled is False
    assert info.analog_enabled is False
    assert info.ecg_enabled is False
    assert info.ppg_enabled is False
    assert info.battery_enabled is False
    assert info.sampling_rate_hz == 102.4
    assert info.session_termination == 'BLE'
    assert info.acc_range_g == 16
    assert info.gyro_range_dps == 2000
    assert info.unix_time_start == 1555598546
    assert info.datetime_start == datetime.datetime(2019, 4, 18, 16, 42, 26)
    assert info.unix_time_stop == 1555598569
    assert info.datetime_stop == datetime.datetime(2019, 4, 18, 16, 42, 49)
    assert info.duration_s == 23
    assert info.num_samples == 2367
    assert info.version_firmware == 'v0.11.5'
    assert info.sync_role == 'disabled'
    assert info.is_synchronised is False
    assert info.sync_group == 0
    assert info.sync_index_start == 0  # not synced
    assert info.sync_index_stop == 0  # not synced
    assert info.mac_address == 'a9:db:22:ac:f7:29'
    assert info.sensor_id == 'f729'
    assert info.sync_address == '19efbeadde'
    assert info.sync_channel == 27

    # System Info
    assert info.sensor_position == 'undefined'
    assert info.has_position_info is False
    assert info.dock_mode_enabled is False
    assert info.motion_interrupt_enabled is False
    assert np.array_equal(info.custom_meta_data, np.zeros(3))
