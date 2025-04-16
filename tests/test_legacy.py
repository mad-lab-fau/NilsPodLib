import json
import tempfile
from contextlib import contextmanager
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest
from packaging.version import Version

from nilspodlib import Dataset
from nilspodlib.exceptions import LegacyWarning, VersionError
from nilspodlib.header import Header
from nilspodlib.legacy import (
    MIN_NON_LEGACY_VERSION,
    _convert_analog_uint8_to_uint16_18_0,
    _convert_sensor_enabled_flag_11_2,
    _fix_little_endian_counter,
    _insert_missing_bytes_11_2,
    _split_sampling_rate_byte_11_2,
    convert_11_2,
    convert_12_0,
    convert_18_0,
    find_conversion_function,
    load_18_0,
)
from nilspodlib.utils import convert_little_endian, get_header_and_data_bytes, get_sample_size_from_header_bytes
from tests.conftest import TEST_LEGACY_DATA_11, TEST_LEGACY_DATA_12, TEST_LEGACY_DATA_16_2, TEST_SESSION_DATA


@contextmanager
def does_not_raise():
    yield


@pytest.fixture()
def simple_session_11_2():
    path = TEST_LEGACY_DATA_11 / "NilsPodX-8433_20190412_172203.bin"
    header, data_bytes = get_header_and_data_bytes(path)
    return path, header, data_bytes


@pytest.fixture()
def simple_session_11_2_json_header():
    return json.load((TEST_LEGACY_DATA_11 / "NilsPodX-8433_20190412_172203_header.json").open("r"))


@pytest.fixture()
def simple_session_11_2_csv():
    df = pd.read_csv(TEST_LEGACY_DATA_11 / "NilsPodX-8433_20190412_172203_data.csv")
    return df.set_index("t")


@pytest.fixture()
def simple_session_12_0():
    path = TEST_LEGACY_DATA_12 / "NilsPodX-7FAD_20190430_0933.bin"
    header, data_bytes = get_header_and_data_bytes(path)
    return path, header, data_bytes


@pytest.fixture()
def simple_session_12_0_json_header():
    return json.load((TEST_LEGACY_DATA_12 / "NilsPodX-7FAD_20190430_0933_header.json").open("r"))


@pytest.fixture()
def simple_session_12_0_csv():
    df = pd.read_csv(TEST_LEGACY_DATA_12 / "NilsPodX-7FAD_20190430_0933_data.csv")
    return df.set_index("t")


@pytest.fixture()
def simple_session_16_2():
    path = TEST_LEGACY_DATA_16_2 / "NilsPodX-6F13_20210109_121625.bin"
    header, data_bytes = get_header_and_data_bytes(path)
    return path, header, data_bytes


@pytest.fixture()
def simple_session_16_2_json_header():
    return json.load((TEST_LEGACY_DATA_16_2 / "NilsPodX-6F13_20210109_121625_header.json").open("r"))


@pytest.fixture()
def simple_session_16_2_csv():
    df = pd.read_csv(TEST_LEGACY_DATA_16_2 / "NilsPodX-6F13_20210109_121625_data.csv")
    return df.set_index("t")


@pytest.fixture()
def simple_session_18_0():
    path = TEST_SESSION_DATA / "NilsPodX-6F13_20210109_162824.bin"
    header, data_bytes = get_header_and_data_bytes(path)
    return path, header, data_bytes


def test_convert_analog_channels_to_uint16(simple_session_16_2):
    _, header, data = simple_session_16_2
    data_converted = _convert_analog_uint8_to_uint16_18_0(data, header)
    assert (len(data_converted) % (get_sample_size_from_header_bytes(header) + 3)) == 0


def test_endian_conversion(simple_session_11_2):
    _, header, data = simple_session_11_2
    sample_size = get_sample_size_from_header_bytes(header)
    data = _fix_little_endian_counter(data, sample_size)
    counter_after = convert_little_endian(np.atleast_2d(data[:, -4:]).T, dtype=float)
    assert np.all(np.diff(counter_after) == 1.0)


def test_sensor_enabled_flag_conversion(simple_session_11_2):
    _, header, _ = simple_session_11_2
    sensors = _convert_sensor_enabled_flag_11_2(header[2])
    enabled_sensors = list()
    for para, val in Header._SENSOR_FLAGS.items():
        if bool(sensors & val[0]) is True:
            enabled_sensors.append(para)

    assert enabled_sensors == ["gyro", "acc"]


def test_insert_missing_bytes(simple_session_11_2):
    _, header, _ = simple_session_11_2
    new_header = _insert_missing_bytes_11_2(header)

    assert len(new_header) == 52


@pytest.mark.parametrize("in_byte,out", [(0x00, (0x00, 0x00)), (0x1A, (0x0A, 0x10)), (0x52, (0x02, 0x50))])
def test_split_sampling_rate(in_byte, out):
    assert _split_sampling_rate_byte_11_2(in_byte) == out


@pytest.mark.parametrize(
    "session, converter",
    [
        ("simple_session_11_2", convert_11_2),
        ("simple_session_12_0", convert_12_0),
        ("simple_session_16_2", convert_18_0),
    ],
)
def test_full_conversion(session, converter, request):
    path = request.getfixturevalue(session)[0]
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        converter(path, tmp.name)
        ds = Dataset.from_bin_file(tmp.name)

    # # Uncomment to update regression files
    # with open(path.parent / (str(path.stem) + '_header.json'), 'w+') as f:
    #     f.write(ds.info.to_json())
    # ds.data_as_df(index='time').to_csv(path.parent / (str(path.stem) + '_data.csv'))

    header = request.getfixturevalue(session + "_json_header")
    csv = request.getfixturevalue(session + "_csv")

    pd.testing.assert_frame_equal(ds.data_as_df(index="time"), csv)

    # Header
    # Check all direct values
    info = ds.info
    assert header == json.loads(info.to_json())


@pytest.mark.parametrize(
    "session, converter, expected",
    [
        ("simple_session_11_2", convert_11_2, does_not_raise()),
        ("simple_session_12_0", convert_12_0, does_not_raise()),
        ("simple_session_16_2", convert_18_0, does_not_raise()),
        ("simple_session_16_2", convert_12_0, pytest.raises(VersionError)),
        ("simple_session_16_2", convert_11_2, pytest.raises(VersionError)),
        ("simple_session_18_0", convert_11_2, pytest.raises(VersionError)),
        ("simple_session_18_0", convert_12_0, pytest.raises(VersionError)),
        ("simple_session_18_0", convert_18_0, pytest.raises(VersionError)),
    ],
)
def test_full_conversion_raises(session, converter, expected, request):
    path = request.getfixturevalue(session)[0]
    with tempfile.NamedTemporaryFile() as tmp:
        with expected:
            converter(path, tmp.name)


def test_strict_version_overwrite(simple_session_16_2):
    with patch("nilspodlib.dataset.find_conversion_function", return_value=load_18_0) as mock_func:
        Dataset.from_bin_file(simple_session_16_2[0], legacy_support="resolve", force_version=Version("12.0.0"))
    mock_func.assert_called_with(Version("12.0.0"), in_memory=True)


@pytest.mark.parametrize(
    "session, converter",
    [
        ("simple_session_11_2", convert_11_2),
        ("simple_session_12_0", convert_12_0),
        ("simple_session_16_2", convert_18_0),
    ],
)
def test_auto_resolve(session, converter, request):
    path = request.getfixturevalue(session)[0]
    ds = Dataset.from_bin_file(path, legacy_support="resolve")

    # # Uncomment to update regression files
    # with open(path.parent / (str(path.stem) + '_header.json'), 'w+') as f:
    #     f.write(ds.info.to_json())
    # ds.data_as_df(index='time').to_csv(path.parent / (str(path.stem) + '_data.csv'))

    header = request.getfixturevalue(session + "_json_header")
    csv = request.getfixturevalue(session + "_csv")

    pd.testing.assert_frame_equal(ds.data_as_df(index="time"), csv)

    # Header
    # Check all direct values
    info = ds.info
    assert header == json.loads(info.to_json())
    if ds.info.strict_version_firmware >= Version("12.0.0"):
        assert len(ds.acc) == ds.info.n_samples


@pytest.mark.parametrize(
    "session, converter",
    [
        ("simple_session_11_2", convert_11_2),
        ("simple_session_12_0", convert_12_0),
        ("simple_session_16_2", convert_18_0),
    ],
)
def test_legacy_error(session, converter, request):
    session = request.getfixturevalue(session)
    path = session[0]

    with pytest.raises(VersionError) as e:
        Dataset.from_bin_file(path)

    assert "legacy support" in str(e.value)

    with pytest.warns(LegacyWarning) as e:
        try:
            Dataset.from_bin_file(path, legacy_support="warn")
        except:
            pass

    assert "legacy support" in str(e[0])

    # test converted session:
    path = session[0]
    with tempfile.NamedTemporaryFile(suffix=".bin") as tmp:
        converter(path, tmp.name)
        Dataset.from_bin_file(tmp.name)


@pytest.mark.parametrize(
    "version, correct_func",
    [
        (Version("0.10.0"), None),
        (Version("0.11.255"), "12_0"),
        (Version("0.12.1"), "12_0"),
        (Version("0.11.1"), None),
        (Version("0.11.2"), "11_2"),
        (Version("0.11.3"), "11_2"),
        (Version("0.14.1"), "18_0"),
        (Version("0.16.2"), "18_0"),
        (Version("0.18.0"), "supported"),
        (Version("255.17.255"), "supported"),
        (MIN_NON_LEGACY_VERSION, "supported"),
    ],
)
def test_find_conversion_function(version, correct_func):
    if correct_func == "supported":
        x, y = 1, 2
        assert find_conversion_function(version, in_memory=False, return_name=False)(x, y) == (x, y)
    elif not correct_func:
        with pytest.raises(VersionError):
            find_conversion_function(version, in_memory=False, return_name=False)
    else:
        func = find_conversion_function(version, in_memory=False, return_name=False)
        assert func.__name__ == "convert_" + correct_func

        func = find_conversion_function(version, in_memory=True, return_name=False)
        assert func.__name__ == "load_" + correct_func

        func = find_conversion_function(version, in_memory=False, return_name=True)
        assert func == "convert_" + correct_func

        func = find_conversion_function(version, in_memory=True, return_name=True)
        assert func == "load_" + correct_func
