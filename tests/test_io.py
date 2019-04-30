import datetime
import json
from unittest.mock import patch

import pandas as pd
from NilsPodLib.datastream import Datastream
from tests.conftest import TEST_SESSION_DATA, TEST_REGRESSION_DATA


class MockDate(datetime.datetime):
    """This feels like a dirty hack, but nothing else seemed to work."""
    mock_tz = 2

    @classmethod
    def fromtimestamp(cls, t, tz=None):
        return super().utcfromtimestamp(t) + datetime.timedelta(hours=cls.mock_tz)


@patch('datetime.datetime', MockDate)
def test_load_simple(dataset_master_simple, dataset_master_simple_json_header, dataset_master_data_csv):
    dataset, path = dataset_master_simple

    # Toplevel Stuff
    assert dataset.path == path
    assert isinstance(dataset.acc, Datastream)
    assert dataset.acc.is_calibrated is False
    assert isinstance(dataset.gyro, Datastream)
    assert dataset.gyro.is_calibrated is False
    assert dataset.baro is None
    assert dataset.mag is None
    assert dataset.battery is None
    assert dataset.analog is None
    assert dataset.ppg is None
    assert dataset.ecg is None

    pd.testing.assert_frame_equal(dataset.data_as_df(), dataset_master_data_csv)

    # Header
    # Check all direct values
    info = dataset.info
    assert dataset_master_simple_json_header == json.loads(info.to_json())
    assert info.utc_datetime_start == datetime.datetime(2019, 4, 30, 7, 33, 12)
    assert info.datetime_start == datetime.datetime(2019, 4, 30, 9, 33, 12)
    assert info.utc_datetime_stop == datetime.datetime(2019, 4, 30, 7, 33, 59)
    assert info.datetime_stop == datetime.datetime(2019, 4, 30, 9, 33, 59)
    assert info.is_synchronised is True
    assert info.has_position_info is False
    assert info.sensor_id == '7fad'

    assert info.sync_index_start == 0
    assert info.sync_index_stop == 0


    # # Uncomment to update regression files
    # with open(TEST_REGRESSION_DATA / (str(path.stem) + '_header.json'), 'w+') as f:
    #     f.write(dataset.info.to_json())
    # dataset.data_as_df().to_csv(TEST_REGRESSION_DATA / (str(path.stem) + '_data.csv'))


def test_sync_info(dataset_slave_simple, dataset_master_simple, dataset_slave_simple_json_header):
    dataset, path = dataset_slave_simple
    master = dataset_master_simple[0]

    # # Uncomment to update regression files
    # with open(TEST_REGRESSION_DATA / (str(path.stem) + '_header.json'), 'w+') as f:
    #     f.write(dataset.info.to_json())

    info = dataset.info
    assert dataset_slave_simple_json_header == json.loads(info.to_json())
    assert info.sync_group == master.info.sync_group
    assert info.sync_address == master.info.sync_address
    assert info.sync_channel == master.info.sync_channel
    assert info.sync_index_stop != 0
