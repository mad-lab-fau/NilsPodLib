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
