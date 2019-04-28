import pytest

from NilsPodLib.session import Session


@pytest.fixture()
def basic_session(dataset_master_simple, dataset_slave_simple):
    return Session([dataset_master_simple[0], dataset_slave_simple[0]])


def test_basic_init(dataset_master_simple, dataset_slave_simple):
    session = Session([dataset_master_simple[0], dataset_slave_simple[0]])
    assert session.datasets == tuple([dataset_master_simple[0], dataset_slave_simple[0]])


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
