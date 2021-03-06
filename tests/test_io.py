import datetime
import json

import pandas as pd
import pytest
import pytz

from nilspodlib.dataset import split_into_sensor_data
from nilspodlib.datastream import Datastream
from nilspodlib.exceptions import InvalidInputFileError
from nilspodlib.header import Header
from nilspodlib.utils import get_header_and_data_bytes, read_binary_uint8


def test_load_simple(dataset_master_simple, dataset_master_simple_json_header, dataset_master_data_csv):
    dataset, path = dataset_master_simple

    # Toplevel Stuff
    assert dataset.path == path
    assert isinstance(dataset.acc, Datastream)
    assert dataset.acc.is_calibrated is False
    assert isinstance(dataset.gyro, Datastream)
    assert dataset.gyro.is_calibrated is False
    assert isinstance(dataset.baro, Datastream)
    assert isinstance(dataset.mag, Datastream)
    assert isinstance(dataset.temperature, Datastream)
    assert isinstance(dataset.analog, Datastream)
    assert dataset.ppg is None
    assert dataset.ecg is None

    pd.testing.assert_frame_equal(dataset.data_as_df(index="time"), dataset_master_data_csv)

    # Header
    # Check all direct values
    info = dataset.info
    assert dataset_master_simple_json_header == json.loads(info.to_json())
    assert info.utc_datetime_start == datetime.datetime(2021, 1, 9, 15, 28, 24, tzinfo=pytz.utc)
    assert info.utc_datetime_stop == datetime.datetime(2021, 1, 9, 15, 28, 31, tzinfo=pytz.utc)
    assert info.is_synchronised is True
    assert info.has_position_info is False
    assert info.sensor_id == "6f13"

    assert info.sync_index_start == 0
    assert info.sync_index_stop == 0


def test_read_binary_sanity_check(dataset_master_simple):
    _, path = dataset_master_simple

    header_bytes, data_bytes = get_header_and_data_bytes(path)
    session_header = Header.from_bin_array(header_bytes[1:])

    sample_size = session_header.sample_size
    n_samples = session_header.n_samples

    data = read_binary_uint8(data_bytes, sample_size, n_samples)

    data = data[:, :-1]

    with pytest.raises(InvalidInputFileError):
        split_into_sensor_data(data, session_header)
