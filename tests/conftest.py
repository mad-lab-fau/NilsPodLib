from pathlib import Path

import pytest
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_DATA = HERE / 'test_data'


@pytest.fixture()
def dataset_master_simple():
    path = TEST_DATA / 'simple_synced_master.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_slave_simple():
    path = TEST_DATA / 'simple_synced_slave.bin'
    return Dataset.from_bin_file(path=path), path
