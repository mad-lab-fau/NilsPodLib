import pytest

from NilsPodLib.session import SyncedSession


@pytest.fixture()
def basic_synced_session(dataset_master_simple, dataset_slave_simple):
    return SyncedSession([dataset_master_simple[0], dataset_slave_simple[0]])


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = SyncedSession([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])

# TODO: Tests for different error cases


def test_master(basic_synced_session):
    assert basic_synced_session.master.info.sensor_id == '9e82'


def test_slaves(basic_synced_session):
    slaves = basic_synced_session.slaves
    assert len(slaves) == 1
    assert slaves[0].info.sensor_id == '9433'


def test_cut_to_sync(basic_synced_session):
    s = basic_synced_session.cut_to_syncregion(only_to_master=True, inplace=False)

    assert s.master.counter[0] == basic_synced_session.datasets[0].counter[0]
    assert s.slaves[0].counter[0] == basic_synced_session.slaves[0].counter[basic_synced_session.info.sync_index_start[1]]
    assert s.master.counter[-1] == basic_synced_session.master.counter[-1]
    assert s.slaves[0].counter[-1] == basic_synced_session.slaves[0].counter[basic_synced_session.info.sync_index_stop[1]]
