import numpy as np
import pytest

from nilspodlib.datastream import Datastream


def test_basic_init():
    data = np.ones((100, 3))
    ds = Datastream(data)
    assert np.array_equal(ds.data, data)
    assert ds.sampling_rate_hz == 1
    assert ds.columns == [0, 1, 2]


def test_full_init():
    data = np.ones((100, 3))
    ds = Datastream(data, 100.0, list("abc"))
    assert np.array_equal(ds.data, data)
    assert ds.sampling_rate_hz == 100.0
    assert ds.columns == ["a", "b", "c"]


@pytest.fixture
def simple_ds():
    data = np.ones((100, 3))
    ds = Datastream(data, 100.0, list("abc"))
    return ds


def test_norm(simple_ds):
    assert np.array_equal(simple_ds.norm(), np.sqrt(3) * np.ones(len(simple_ds.data)))


def test_normalize(simple_ds):
    simple_ds.data *= 2
    assert np.array_equal(simple_ds.normalize().data, np.ones((len(simple_ds.data), 3)))


def test_cut(simple_ds):
    simple_ds.data = np.arange(100.0)
    c = simple_ds.cut(10, 90, 2)
    assert c.data[0] == 10.0
    assert c.data[1] == 12.0
    assert c.data[-1] == 88


@pytest.mark.parametrize("factor", [2, 4, 5])
def test_downsample(simple_ds, factor):
    simple_ds.data = np.arange(100.0)
    d = simple_ds.downsample(factor)
    assert len(d.data) == len(simple_ds.data) / factor
    assert d.sampling_rate_hz == simple_ds.sampling_rate_hz / factor


def test_len(simple_ds):
    assert len(simple_ds) == len(simple_ds.data)


def test_columns():
    ds = Datastream(np.zeros((100, 3)))

    assert ds.columns == [0, 1, 2]

    ds = Datastream(np.zeros((100, 2)))

    assert ds.columns == [0, 1]

    ds = Datastream(np.zeros((100, 3)), sensor_type="acc")

    assert ds.columns == ["acc_x", "acc_y", "acc_z"]

    ds = Datastream(np.zeros((100, 3)), columns=["col1", "col2", "col3"])

    assert ds.columns == ["col1", "col2", "col3"]


def test_unit():
    ds = Datastream(np.zeros((100, 3)))

    assert ds.unit == "a.u."
    ds.is_factory_calibrated = True
    assert ds.unit == "a.u."

    ds = Datastream(np.zeros((100, 3)), sensor_type="acc")

    assert ds.unit == "a.u."
    ds.is_factory_calibrated = True
    assert ds.unit == "m/s^2"
    ds.is_calibrated = True
    with pytest.raises(ValueError):
        assert ds.unit == "something"
    ds.calibrated_unit = "m/s^2"
    assert ds.unit == "m/s^2"

    ds = Datastream(np.zeros((100, 3)), calibrated_unit="test", sensor_type="acc")

    assert ds.unit == "a.u."
    ds.is_calibrated = True
    assert ds.unit == "test"
    ds.is_factory_calibrated = True
    # Factory cal flag is ignored, as normal calibration is more important
    assert ds.unit == "test"

    ds = Datastream(np.zeros((100, 3)), sensor_type="not_sensor")

    assert ds.unit == "a.u."
    ds.is_factory_calibrated = True
    assert ds.unit == "a.u."
