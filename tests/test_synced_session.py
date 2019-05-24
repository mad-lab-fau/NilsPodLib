import pytest

import numpy as np
from NilsPodLib.session import SyncedSession


@pytest.fixture()
def basic_synced_session(dataset_master_simple, dataset_slave_simple, dataset_analog_simple):
    return SyncedSession([dataset_master_simple[0], dataset_slave_simple[0], dataset_analog_simple[0]])


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = SyncedSession([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])

@pytest.mark.parametrize('para', [
    'sync_group',
    'sync_channel'
])
def test_validate_sync_group(dataset_master_simple, dataset_slave_simple, para):
    ds1 = dataset_slave_simple[0]
    ds2 = dataset_master_simple[0]

    setattr(ds1.info, para, getattr(ds2.info, para) + 1)
    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert 'sync_group' in str(e)

    ds1.info.sync_address = ds2.info.sync_address + 'a'
    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert 'sync_group' in str(e)


def test_start_end_validation(dataset_master_simple, dataset_slave_simple):
    ds1 = dataset_slave_simple[0]
    ds2 = dataset_master_simple[0]

    ds1.info.utc_start = ds2.info.utc_start - 10000
    ds1.info.utc_stop = ds2.info.utc_stop - 5000

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert 'overlapping time period' in str(e)


@pytest.mark.parametrize('roles, err', [
    (('master', 'master', 'slave'), 'exactly 1 master'),
    (('slave', 'slave', 'slave'), 'exactly 1 master'),
    (('master', 'slave', 'disabled'), 'either slave or master')
])
def test_two_master_validation(dataset_master_simple, dataset_slave_simple, dataset_analog_simple, roles, err):
    ds1 = dataset_slave_simple[0]
    ds2 = dataset_master_simple[0]
    ds3 = dataset_analog_simple[0]

    ds1.info.sync_role = roles[0]
    ds2.info.sync_role = roles[1]
    ds3.info.sync_role = roles[2]

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2, ds3])

    assert err in str(e)


def test_validate_sampling_rate(dataset_master_simple, dataset_slave_simple):
    ds1 = dataset_slave_simple[0]
    ds2 = dataset_master_simple[0]

    ds1.info.sampling_rate_hz = ds2.info.sampling_rate_hz + 5

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert 'same sampling rate' in str(e)


def test_disable_validation(dataset_master_simple, dataset_slave_simple):
    ds1 = dataset_slave_simple[0]
    ds2 = dataset_master_simple[0]

    ds1.info.sync_address = ds2.info.sync_address + 'a'
    with pytest.raises(ValueError):
        SyncedSession([ds1, ds2])

    SyncedSession.VALIDATE_ON_INIT = False
    SyncedSession([ds1, ds2])


# TODO: Tests for different error cases


def test_master(basic_synced_session, dataset_master_simple):
    assert basic_synced_session.master.info.sensor_id == dataset_master_simple[0].info.sensor_id


def test_slaves(basic_synced_session, dataset_slave_simple, dataset_analog_simple):
    slaves = basic_synced_session.slaves
    slave_ids = [d.info.sensor_id for d in slaves]
    assert len(slaves) == 2
    assert dataset_slave_simple[0].info.sensor_id in slave_ids
    assert dataset_analog_simple[0].info.sensor_id in slave_ids


def test_cut_to_sync_only_master_with_end(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(only_to_master=True, end=True, inplace=False)

    assert s.master.counter[0] == basic_synced_session.master.counter[0]
    assert s.slaves[0].counter[0] == basic_synced_session.slaves[0].counter[
        basic_synced_session.slaves[0].info.sync_index_start]
    assert s.slaves[1].counter[0] == basic_synced_session.slaves[1].counter[
        basic_synced_session.slaves[1].info.sync_index_start]
    assert s.master.counter[-1] == basic_synced_session.master.counter[-1]
    assert s.slaves[0].counter[-1] == basic_synced_session.slaves[0].counter[
        basic_synced_session.slaves[0].info.sync_index_stop]
    assert s.slaves[1].counter[-1] == basic_synced_session.slaves[1].counter[
        basic_synced_session.slaves[1].info.sync_index_stop]


def test_cut_to_sync_only_master_without_end(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(only_to_master=True, end=False, inplace=False)

    assert s.master.counter[0] == basic_synced_session.master.counter[0]
    assert s.slaves[0].counter[0] == basic_synced_session.slaves[0].counter[
        basic_synced_session.slaves[0].info.sync_index_start]
    assert s.slaves[1].counter[0] == basic_synced_session.slaves[1].counter[
        basic_synced_session.slaves[1].info.sync_index_start]
    assert s.master.counter[-1] == basic_synced_session.master.counter[-1]
    assert s.slaves[0].counter[-1] == basic_synced_session.slaves[0].counter[-1]
    assert s.slaves[1].counter[-1] == basic_synced_session.slaves[1].counter[-1]


def test_cut_to_sync_with_end(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(end=True, inplace=False)

    start = np.array([d.counter[d.info.sync_index_start] for d in basic_synced_session.slaves]).max()
    stop = np.array([d.counter[d.info.sync_index_stop] + 1 for d in basic_synced_session.slaves]).min()

    for d in s.datasets:
        assert len(d.counter) == stop - start
        assert np.array_equal(d.counter, s.master.counter)


def test_cut_to_sync_without_end(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(end=False, inplace=False)

    start = np.array([d.counter[d.info.sync_index_start] for d in basic_synced_session.slaves]).max()
    length = np.array([len(d.counter) - d.info.sync_index_start - 1 for d in basic_synced_session.slaves]).min()

    for d in s.datasets:
        assert d.counter[0] == start
        assert len(d.counter) == length
        assert np.array_equal(d.counter, s.master.counter)


def test_cut_to_sync_warn(basic_synced_session):
    s = basic_synced_session

    with pytest.warns(None) as rec:
        s.cut_to_syncregion(end=False)

    assert len(rec) == 0

    thres = 0
    with pytest.warns(UserWarning) as rec:
        s.cut_to_syncregion(end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([d.info.sensor_id for d in s.slaves]) in str(rec[0])

    thres = 30
    s.slaves[0].info.sync_index_stop -= int(30 * s.slaves[0].info.sampling_rate_hz)
    with pytest.warns(UserWarning) as rec:
        s.cut_to_syncregion(end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([s.slaves[0].info.sensor_id]) in str(rec[0])

    thres = 30
    s.slaves[0].info.sync_index_stop -= int(30 * s.slaves[0].info.sampling_rate_hz)
    with pytest.warns(UserWarning) as rec:
        s.cut_to_syncregion(end=False, warn_thres=thres, only_to_master=True)

    assert len(rec) == 1

    with pytest.warns(None) as rec:
        s.cut_to_syncregion(end=False, warn_thres=None)

    assert len(rec) == 0



