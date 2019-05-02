import copy

import numpy as np
import pandas as pd
import pytest


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

    df = ds.data_as_df()
    assert len(df.columns) == 6

    df = ds.data_as_df(datastreams=('gyro',))
    assert len(df.columns) == 3


def test_imu_data_as_df(dataset_master_simple):
    ds = dataset_master_simple[0]

    df = ds.imu_data_as_df()

    assert len(df.columns) == 6
    pd.testing.assert_frame_equal(ds.imu_data_as_df(), ds.data_as_df(datastreams=('acc', 'gyro')))


def test_cut(dataset_master_simple):
    ds = dataset_master_simple[0]
    # Add fake counter and fake acc to easily check cut
    ds.counter = np.arange(len(ds.counter))

    ds_new = ds.cut(0, 100)
    assert ds_new.counter[0] == 0.
    assert ds_new.counter[-1] == 99.
    assert len(ds_new.counter) == 100

    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[0])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[99])
    assert len(ds_new.acc.data) == 100

    # inplace
    ds_new = copy.deepcopy(ds)
    ds_new.cut(0, 100, inplace=True)
    assert ds_new.counter[0] == 0.
    assert ds_new.counter[-1] == 99.
    assert len(ds_new.counter) == 100

    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[0])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[99])
    assert len(ds_new.acc.data) == 100


def test_cut_to_sync_master(dataset_master_simple):
    ds = dataset_master_simple[0]
    ds_new = ds.cut_to_syncregion()

    assert np.array_equal(ds.counter, ds_new.counter)


def test_cut_to_sync_slave_with_end(dataset_slave_simple):
    ds = dataset_slave_simple[0]
    ds_new = ds.cut_to_syncregion(end=True)

    assert ds_new.counter[0] == ds.counter[ds.info.sync_index_start]
    assert ds_new.counter[-1] == ds.counter[ds.info.sync_index_stop]
    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[ds.info.sync_index_start])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[ds.info.sync_index_stop])


def test_cut_to_sync_slave_without_end(dataset_slave_simple):
    ds = dataset_slave_simple[0]
    ds_new = ds.cut_to_syncregion(end=False)

    assert ds_new.counter[0] == ds.counter[ds.info.sync_index_start]
    assert ds_new.counter[-1] == ds.counter[-1]
    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[ds.info.sync_index_start])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[-1])


def test_cut_to_sync_warning(dataset_slave_simple):
    ds = dataset_slave_simple[0]

    with pytest.warns(None) as rec:
        ds.cut_to_syncregion(end=False)

    assert len(rec) == 0

    thres = 0
    with pytest.warns(UserWarning) as rec:
        ds.cut_to_syncregion(end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])

    thres = 30
    ds.info.sync_index_stop -= int(30 * ds.info.sampling_rate_hz)
    with pytest.warns(UserWarning) as rec:
        ds.cut_to_syncregion(end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])

    with pytest.warns(None) as rec:
        ds.cut_to_syncregion(end=False, warn_thres=None)

    assert len(rec) == 0


def test_cut_to_counter_value(dataset_master_simple):
    ds = dataset_master_simple[0]
    # Add fake counter and fake acc to easily check cut
    ds.counter = np.arange(len(ds.counter)) + 10

    ds_new = ds.cut_counter_val(10, 110)
    assert ds_new.counter[0] == 10.
    assert ds_new.counter[-1] == 109.
    assert len(ds_new.counter) == 100

    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[0])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[99])
    assert len(ds_new.acc.data) == 100

    # inplace
    ds_new = copy.deepcopy(ds)
    ds_new.cut_counter_val(10, 110, inplace=True)
    assert ds_new.counter[0] == 10.
    assert ds_new.counter[-1] == 109.
    assert len(ds_new.counter) == 100

    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[0])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[99])
    assert len(ds_new.acc.data) == 100


def test_datastreams():
    pass
