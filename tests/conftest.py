import json
from pathlib import Path

import pandas as pd
import pytest

from nilspodlib.dataset import Dataset

HERE = Path(__file__).parent
TEST_LEGACY_DATA_11 = HERE / "test_data/11_2_legacy_sample"
TEST_LEGACY_DATA_12 = HERE / "test_data/12_0_legacy_sample"
TEST_LEGACY_DATA_14_1 = HERE / "test_data/14_1_legacy_sample"
TEST_LEGACY_DATA_16_2 = HERE / "test_data/16_2_legacy_sample"
TEST_SESSION_DATA = HERE / "test_data/18_0_sample"
TEST_SYNCED_DATA = HERE / "test_data/synced_sample_session"


def _dataset_master_simple():
    path = TEST_SESSION_DATA / "NilsPodX-6F13_20210109_152726.bin"
    return Dataset.from_bin_file(path=path), path


@pytest.fixture()
def dataset_master_simple():
    return _dataset_master_simple()


@pytest.fixture()
def dataset_master_simple_json_header():
    return json.load((TEST_SESSION_DATA / "NilsPodX-6F13_20210109_162824_header.json").open("r"))


@pytest.fixture()
def dataset_master_data_csv():
    df = pd.read_csv(TEST_SESSION_DATA / "NilsPodX-6F13_20210109_162824_data.csv")
    return df.set_index("t")


@pytest.fixture()
def dataset_synced():
    master = TEST_SYNCED_DATA / "NilsPodX-7FAD_20190430_0933.bin"
    slave1 = TEST_SYNCED_DATA / "NilsPodX-922A_20190430_0933.bin"
    slave2 = TEST_SYNCED_DATA / "NilsPodX-323C_20190430_0933.bin"
    return {
        "master": (Dataset.from_bin_file(master, ), master),
        "slave1": (Dataset.from_bin_file(slave1), slave1),
        "slave2": (Dataset.from_bin_file(slave2), slave2),
    }


# # Uncomment to update regression files
# dataset, path = _dataset_master_simple()
# with open(path.parent / (str(path.stem) + "_header.json"), "w+") as f:
#     f.write(dataset.info.to_json())
# dataset.data_as_df(index="time").to_csv(path.parent / (str(path.stem) + "_data.csv"))
