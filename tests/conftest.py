import json
from pathlib import Path
import pandas as pd

import pytest
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_LEGACY_DATA = HERE / 'test_data/11_2_legacy_sample'
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
def dataset_analog_simple():
    path = TEST_SESSION_DATA / 'NilsPodX-323C_20190430_0933.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_master_simple_json_header():
    return json.load((TEST_REGRESSION_DATA / 'NilsPodX-7FAD_20190430_0933_header.json').open('r'))


@pytest.fixture()
def dataset_slave_simple_json_header():
    return json.load((TEST_REGRESSION_DATA / 'NilsPodX-922A_20190430_0933_header.json').open('r'))


@pytest.fixture()
def dataset_master_data_csv():
    df = pd.read_csv(TEST_REGRESSION_DATA / 'NilsPodX-7FAD_20190430_0933_data.csv')
    return df.set_index('t')
