import datetime
import json
from unittest.mock import patch

import pandas as pd

from NilsPodLib.datastream import Datastream


def test_load_simple(dataset_master_simple, dataset_master_simple_json_header, dataset_master_data_csv):
    dataset, path = dataset_master_simple

    # # Uncomment to update regression files
    # with open(path.parent / (str(path.stem) + '_header.json'), 'w+') as f:
    #     f.write(dataset.info.to_json())
    # dataset.data_as_df(index='time').to_csv(path.parent / (str(path.stem) + '_data.csv'))

    # Toplevel Stuff
    assert dataset.path == path
    assert isinstance(dataset.acc, Datastream)
    assert dataset.acc.is_calibrated is False
    assert isinstance(dataset.gyro, Datastream)
    assert dataset.gyro.is_calibrated is False
    assert dataset.baro is None
    assert dataset.mag is None
    assert dataset.temperature is None
    assert dataset.analog is None
    assert dataset.ppg is None
    assert dataset.ecg is None

    pd.testing.assert_frame_equal(dataset.data_as_df(index='time'), dataset_master_data_csv)

    # Header
    # Check all direct values
    info = dataset.info
    assert dataset_master_simple_json_header == json.loads(info.to_json())
    assert info.utc_datetime_start == datetime.datetime(2019, 5, 22, 12, 51, 21)
    assert info.utc_datetime_stop == datetime.datetime(2019, 5, 22, 12, 52, 11)
    assert info.is_synchronised is True
    assert info.has_position_info is False
    assert info.sensor_id == '9433'

    assert info.sync_index_start == 0
    assert info.sync_index_stop == 0
