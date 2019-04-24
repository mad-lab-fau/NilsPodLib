import pytest
from NilsPodLib.dataset import Dataset
from pathlib import Path
import numpy as np
from numpy.testing import assert_almost_equal

HERE = Path(__file__).parent
TEST_DATA = HERE / 'test_data'

@pytest.fixture()
def dataset():
    path = TEST_DATA / 'simple_synced_master.bin'
    return Dataset(path=path)


def test_size(dataset):
    # TODO: Why is the actual dataset shorter than the reported number of samples?
    # assert dataset.size == dataset.info.num_samples
    assert len(dataset.acc) == dataset.size
    assert len(dataset.counter) == dataset.size


def test_counter(dataset):
    """Counter should be monotonic and be in the correct range for sample since midnight."""
    assert np.all(np.diff(dataset.counter)) == 1
    t = dataset.info.datetime_start
    seconds_on_date = t.hour*3600 + t.minute * 60 + t.second
    assert int(dataset.counter[0] / dataset.info.sampling_rate_hz) == int(seconds_on_date)


def test_datastreams():
    pass
