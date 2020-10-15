import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from NilsPodLib.header import HeaderFields
from NilsPodLib.session import Session


@pytest.fixture()
def basic_session(dataset_synced):
    return Session([dataset_synced["master"][0], dataset_synced["slave1"][0]])


def test_basic_init(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]
    session = Session([ds1, ds2])
    assert session.datasets == tuple([ds1, ds2])


def test_init_from_file_paths(dataset_synced):
    ds1 = dataset_synced["master"]
    ds2 = dataset_synced["slave1"]
    session = Session.from_file_paths([ds1[1], ds2[1]])
    assert ds1[0].info.sensor_id in session.info.sensor_id
    assert ds2[0].info.sensor_id in session.info.sensor_id
    assert len(session.datasets) == 2


def test_init_from_folder(dataset_synced):
    session = Session.from_folder_path(Path(dataset_synced["master"][1]).parent)
    assert dataset_synced["master"][0].info.sensor_id in session.info.sensor_id
    assert dataset_synced["slave1"][0].info.sensor_id in session.info.sensor_id
    assert dataset_synced["slave2"][0].info.sensor_id in session.info.sensor_id
    assert len(session.datasets) == 3


def test_init_from_folder_empty(dataset_synced):
    with tempfile.TemporaryDirectory() as folder:
        with pytest.raises(ValueError):
            Session.from_folder_path(folder)


@pytest.mark.parametrize(
    "name",
    [
        *HeaderFields()._header_fields,
        "duration_s",  # As example for a property
    ],
)
def test_info_access(name, basic_session):
    session = basic_session

    assert getattr(session.info, name) == tuple((getattr(d.info, name) for d in session.datasets))


def test_info_write(basic_session):
    with pytest.raises(NotImplementedError):
        basic_session.info.test = 4


def test_info_get_method(basic_session):
    with pytest.raises(ValueError) as e:
        basic_session.info.from_json("test")

    assert "from_json" in str(e.value)


def test_dataset_attr_access(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]
    session = Session([ds1, ds2])
    assert session.acc == (ds1.acc, ds2.acc)


def test_dataset_func_call(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    session = Session([ds1, ds2])
    pd.testing.assert_frame_equal(session.data_as_df()[0], ds1.data_as_df())
    pd.testing.assert_frame_equal(session.data_as_df()[1], ds2.data_as_df())


def test_dataset_func_call_dataset_return(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    session = Session([ds1, ds2])
    return_val = session.cut(0, 10)
    assert isinstance(return_val, Session)
    assert np.array_equal(return_val.datasets[0].acc.data, ds1.cut(0, 10).acc.data)
    assert np.array_equal(return_val.datasets[1].acc.data, ds2.cut(0, 10).acc.data)


def test_get_dataset_by_id(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    session = Session([ds1, ds2])

    assert session.get_dataset_by_id(ds1.info.sensor_id).info.sensor_id == ds1.info.sensor_id
