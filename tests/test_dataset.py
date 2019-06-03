import copy
from datetime import datetime

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
    assert np.array_equal(df.index.values, np.arange(len(ds.counter)))

    df = ds.data_as_df(datastreams=('gyro',))
    assert len(df.columns) == 3

    df = ds.data_as_df(index='counter')
    assert np.array_equal(df.index, ds.counter)

    df = ds.data_as_df(index='time')
    assert np.array_equal(df.index.values, ds.time_counter)

    df = ds.data_as_df(index='utc')
    assert np.array_equal(df.index.values, ds.utc_counter)

    df = ds.data_as_df(index='utc_datetime')
    assert np.array_equal(df.index.values, ds.utc_datetime_counter)


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
    assert ds_new.counter[-1] == ds.counter[ds.info.sync_index_stop - 1]
    assert np.array_equal(ds_new.acc.data[0], ds.acc.data[ds.info.sync_index_start])
    assert np.array_equal(ds_new.acc.data[-1], ds.acc.data[ds.info.sync_index_stop - 1])


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


def test_utc_counter(dataset_master_simple):
    ds = dataset_master_simple[0]

    assert int(ds.utc_counter[0]) == int(ds.info.utc_start)
    # As the last page is not transmitted, the values will not be exactly the same, but they should be close
    assert int(ds.utc_counter[-1] // 10) == int(ds.info.utc_stop // 10)
    assert len(ds.utc_counter) == len(ds.counter)


def test_utc_datetime_counter(dataset_master_simple):
    ds = dataset_master_simple[0]

    assert ds.utc_datetime_counter[0].astype('datetime64[s]') == np.datetime64(ds.info.utc_datetime_start)
    # # As the last page is not transmitted, the values will not be exactly the same, but they should be close
    assert np.abs(ds.utc_datetime_counter[-1].astype('datetime64[s]') - np.datetime64(ds.info.utc_datetime_stop).astype('datetime64[s]')) <= np.timedelta64(2, 's')
    assert len(ds.utc_datetime_counter) == len(ds.counter)


@pytest.mark.parametrize('factor', [2, 4, 5])
def test_downsample(dataset_master_simple, factor):
    ds = dataset_master_simple[0]
    ds.data = np.arange(100.)
    s = ds.downsample(factor)
    for k, d in s.datastreams:
        assert len(d.data) == int(np.ceil(len(getattr(ds, k).data) / factor))
        assert d.sampling_rate_hz == getattr(ds, k).sampling_rate_hz / factor
    assert len(s.counter) == int(np.ceil(len(ds.counter) / factor))
