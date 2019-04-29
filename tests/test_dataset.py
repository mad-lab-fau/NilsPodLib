import copy

import numpy as np
import pandas as pd


def test_size(dataset_master_simple):
    dataset, _ = dataset_master_simple
    assert len(dataset.acc) == dataset.size
    assert len(dataset.counter) == dataset.size


def test_counter(dataset_master_simple):
    """Counter should be monotonic and be in the correct range for sample since midnight."""
    dataset, _ = dataset_master_simple
    assert np.all(np.diff(dataset.counter)) == 1
    t = dataset.info.utc_datetime_start
    seconds_on_date = t.hour*3600 + t.minute * 60 + t.second
    assert int(dataset.counter[0] / dataset.info.sampling_rate_hz) == int(seconds_on_date)


def test_data_as_df(dataset_master_simple):
    ds = dataset_master_simple[0]
    # Add fake gyro
    ds.gyro = copy.deepcopy(ds.acc)
    ds.info.enabled_sensors = ('gyro', *ds.ACTIVE_SENSORS)

    df = ds.data_as_df()
    assert len(df.columns) == 6

    df = ds.data_as_df(datastreams=('gyro',))
    assert len(df.columns) == 3


def test_imu_data_as_df(dataset_master_simple):
    ds = dataset_master_simple[0]
    # Add fake gyro
    ds.gyro = copy.deepcopy(ds.acc)
    ds.info.enabled_sensors = ('gyro', *ds.ACTIVE_SENSORS)

    df = ds.imu_data_as_df()

    assert len(df.columns) == 6
    pd.testing.assert_frame_equal(ds.imu_data_as_df(), ds.data_as_df(datastreams=('acc', 'gyro')))


def test_datastreams():
    pass
