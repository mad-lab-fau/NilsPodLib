import json
from pathlib import Path
import pandas as pd

import pytest
from NilsPodLib.dataset import Dataset

HERE = Path(__file__).parent
TEST_LEGACY_DATA_11 = HERE / 'test_data/11_2_legacy_sample'
TEST_LEGACY_DATA_12 = HERE / 'test_data/12_0_legacy_sample'
TEST_SESSION_DATA = HERE / 'test_data/14_0_sample'
TEST_SYNCED_DATA  = HERE / 'test_data/synced_sample_session'


@pytest.fixture()
def dataset_master_simple():
    path = TEST_SESSION_DATA / 'NilsPodX-9433_20190522_1451.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_slave_simple():
    path = TEST_SESSION_DATA / 'NilsPodX-E0EF_20190522_1451.bin'
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_master_simple_json_header():
    return json.load((TEST_SESSION_DATA / 'NilsPodX-9433_20190522_1451_header.json').open('r'))


@pytest.fixture()
def dataset_slave_simple_json_header():
    return json.load((TEST_SESSION_DATA / 'NilsPodX-E0EF_20190522_1451_header.json').open('r'))

@pytest.fixture()
def dataset_master_data_csv():
    df = pd.read_csv(TEST_SESSION_DATA / 'NilsPodX-9433_20190522_1451_data.csv')
    return df.set_index('t')

@pytest.fixture()
def dataset_synced():
    master = TEST_SYNCED_DATA / 'NilsPodX-7FAD_20190430_0933.bin'
    slave1 = TEST_SYNCED_DATA / 'NilsPodX-922A_20190430_0933.bin'
    slave2 = TEST_SYNCED_DATA / 'NilsPodX-323C_20190430_0933.bin'
    return {
        'master': (Dataset.from_bin_file(master), master),
        'slave1': (Dataset.from_bin_file(slave1), slave1),
        'slave2': (Dataset.from_bin_file(slave2), slave2)
            }
