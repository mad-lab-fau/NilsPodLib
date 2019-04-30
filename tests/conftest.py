import json
from pathlib import Path

import pytest
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_SESSION_DATA = HERE / 'test_data/sample_session'
TEST_REGRESSION_DATA = HERE / 'test_data/sample_session_regression'


@pytest.fixture()
def dataset_master_simple():
    path = TEST_SESSION_DATA / 'NilsPodX-7FAD_20190430_0933.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_slave_simple():
    path = TEST_SESSION_DATA / 'NilsPodX-922A_20190430_0933.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_master_simple_json_header():
    return json.load((TEST_REGRESSION_DATA / 'NilsPodX-7FAD_20190430_0933_header.json').open('r'))


@pytest.fixture()
def dataset_slave_simple_json_header():
    return json.load((TEST_REGRESSION_DATA / 'NilsPodX-922A_20190430_0933_header.json').open('r'))


