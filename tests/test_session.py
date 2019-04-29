import pandas as pd
import numpy as np
import pytest
from pathlib import Path

from NilsPodLib.dataset import ProxyDataset
from NilsPodLib.session import Session


@pytest.fixture()
def basic_session(dataset_master_simple, dataset_slave_simple):
    return Session([dataset_master_simple[0], dataset_slave_simple[0]])


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets._datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])
    assert isinstance(session.datasets, ProxyDataset)


def test_init_from_file_paths(dataset_master_simple, dataset_slave_simple):
    session = Session.from_file_paths([dataset_master_simple[1], dataset_slave_simple[1]])
    assert dataset_master_simple[0].info.sensor_id in session.info.sensor_id
    assert dataset_slave_simple[0].info.sensor_id in session.info.sensor_id
    assert len(session.datasets._datasets) == 2
    assert isinstance(session.datasets, ProxyDataset)


def test_init_from_folder(dataset_master_simple, dataset_slave_simple):
    session = Session.from_folder_path(Path(dataset_master_simple[0].path).parent)
    assert dataset_master_simple[0].info.sensor_id in session.info.sensor_id
    assert dataset_slave_simple[0].info.sensor_id in session.info.sensor_id
    assert len(session.datasets._datasets) == 2
    assert isinstance(session.datasets, ProxyDataset)


def test_info_access(basic_session):
    session = basic_session

    assert session.info.sensor_id == ('9e82', '9433')  # Test a property
    assert session.info.enabled_sensors == (('acc',), ('acc',))  # Test complex datatype
    assert session.info.enabled_sensors == (('acc',), ('acc',))


def test_info_write(basic_session):
    with pytest.raises(NotImplementedError):
        basic_session.info.test = 4


def test_info_get_method(basic_session):
    with pytest.raises(ValueError) as e:
        basic_session.info.from_json('test')

    assert 'from_json' in str(e)


def test_dataset_access(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    assert [d for d in session.datasets] == [dataset_master_simple[0], dataset_slave_simple[0]]
    assert session.datasets[0] == dataset_master_simple[0]
    assert session.datasets[1] == dataset_slave_simple[0]


def test_dataset_attr_access(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets.acc == (dataset_master_simple[0].acc, dataset_slave_simple[0].acc)


def test_dataset_func_call(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    pd.testing.assert_frame_equal(session.datasets.data_as_df()[0], dataset_master_simple[0].data_as_df())
    pd.testing.assert_frame_equal(session.datasets.data_as_df()[1], dataset_slave_simple[0].data_as_df())


def test_dataset_func_call_dataset_return(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    return_val = session.datasets.cut(0, 10)
    assert isinstance(return_val, ProxyDataset)
    assert np.array_equal(return_val[0].acc.data, dataset_master_simple[0].cut(0, 10).acc.data)
    assert np.array_equal(return_val[1].acc.data, dataset_slave_simple[0].cut(0, 10).acc.data)
