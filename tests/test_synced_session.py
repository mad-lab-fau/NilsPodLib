import pytest

import numpy as np
from NilsPodLib.session import SyncedSession


@pytest.fixture()
def basic_synced_session(dataset_master_simple, dataset_slave_simple, dataset_analog_simple):
    return SyncedSession([dataset_master_simple[0], dataset_slave_simple[0], dataset_analog_simple[0]])


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = SyncedSession([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])


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
    s = basic_synced_session.cut_to_syncregion(only_to_master=True, end=True, inplace=False)

    assert s.master.counter[0] == basic_synced_session.master.counter[0]
    assert s.slaves[0].counter[0] == basic_synced_session.slaves[0].counter[
        basic_synced_session.slaves[0].info.sync_index_start]
    assert s.slaves[1].counter[0] == basic_synced_session.slaves[1].counter[
        basic_synced_session.slaves[1].info.sync_index_start]
    assert s.master.counter[-1] == basic_synced_session.master.counter[-1]
    assert s.slaves[0].counter[-1] == basic_synced_session.slaves[0].counter[-1]
    assert s.slaves[1].counter[-1] == basic_synced_session.slaves[1].counter[-1]


def test_cut_to_sync(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(inplace=False)

    start = np.array([d.counter[d.info.sync_index_start] for d in s.slaves]).max()
    stop = np.array([d.counter[d.info.sync_index_stop] for d in s.slaves]).min()

    for d in s.datasets:
        assert len(d.counter) == stop - start
        assert np.array_equal(d.counter, s.master.counter)


