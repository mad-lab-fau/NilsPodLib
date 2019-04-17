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
    assert np.array_equal(simple_ds.normalize(), np.ones((len(simple_ds.data), 3)))

