import pytest

from NilsPodLib.datastream import Datastream
import numpy as np


def test_basic_init():
    data = np.ones((100, 3))
    ds = Datastream(data)
    assert np.array_equal(ds.data, data)
    assert ds.sampling_rate_hz == 1
    assert ds.columns == [0, 1, 2]


def test_full_init():
    data = np.ones((100, 3))
    ds = Datastream(data, 100., list('abc'))
    assert np.array_equal(ds.data, data)
    assert ds.sampling_rate_hz == 100.
    assert ds.columns == ['a', 'b', 'c']


@pytest.fixture()
def simple_ds():
    data = np.ones((100, 3))
    ds = Datastream(data, 100., list('abc'))
    return ds


def test_norm(simple_ds):
    assert np.array_equal(simple_ds.norm(), np.sqrt(3) * np.ones(len(simple_ds.data)))


def test_normalize(simple_ds):
    simple_ds.data *= 2
    assert np.array_equal(simple_ds.normalize().data, np.ones((len(simple_ds.data), 3)))


def test_cut(simple_ds):
    simple_ds.data = np.arange(100.)
    c = simple_ds.cut(10, 90, 2)
    assert c.data[0] == 10.
    assert c.data[1] == 12.
    assert c.data[-1] == 88


@pytest.mark.parametrize('factor', [2, 4, 5])
def test_downsample(simple_ds, factor):
    simple_ds.data = np.arange(100.)
    d = simple_ds.downsample(factor)
    assert len(d.data) == len(simple_ds.data) / factor
    assert d.sampling_rate_hz == simple_ds.sampling_rate_hz / factor


def test_len(simple_ds):
    assert len(simple_ds) == len(simple_ds.data)
