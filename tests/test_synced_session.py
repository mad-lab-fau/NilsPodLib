from NilsPodLib.dataset import ProxyDataset
from NilsPodLib.session import SyncedSession


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = SyncedSession([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets._datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])
    assert isinstance(session.datasets, ProxyDataset)