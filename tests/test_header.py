from datetime import timedelta
from distutils.version import StrictVersion

import pytest

from nilspodlib.header import Header


def test_header_init_warning():
    with pytest.warns(UserWarning) as warn:
        Header(not_an_kwarg=None)

    assert len(warn) == 1
    assert "not_an_kwarg" in warn[0].message.args[0]


def test_json_roundtrip_simple():
    paras = dict(sampling_rate_hz=100, enabled_sensors=("acc",), sync_role="master", custom_meta_data=(1, 2, 3, 4))
    header = Header(**paras)
    new_header = Header.from_json(header.to_json())
    for k in paras.keys():
        assert getattr(header, k) == getattr(new_header, k)


def test_json_roundtrip(dataset_master_simple):
    header = dataset_master_simple[0].info
    new_header = Header.from_json(header.to_json())
    for k in header._header_fields:
        assert getattr(header, k) == getattr(new_header, k)


def test_start_midnight_daytime(dataset_master_simple):
    header = dataset_master_simple[0].info

    start = header.utc_datetime_start_day_midnight

    assert start.hour == 0
    assert start.minute == 0
    assert start.second == 0
    assert start.date() == header.utc_datetime_start.date()


def test_strict_version():
    h = Header()
    h.version_firmware = "v1.0.5"

    assert h.strict_version_firmware == StrictVersion("1.0.5")


def test_timezone_conversion(dataset_master_simple):
    header = dataset_master_simple[0].info
    header.timezone = "Europe/Berlin"

    berlin_start_time = header.local_datetime_start
    berlin_stop_time = header.local_datetime_stop

    assert berlin_start_time.hour - header.utc_datetime_start.hour == 1
    assert berlin_stop_time.hour - header.utc_datetime_stop.hour == 1

    header.timezone = "Europe/London"

    london_start_time = header.local_datetime_start
    london_stop_time = header.local_datetime_stop

    assert london_start_time.hour - header.utc_datetime_start.hour == 0
    assert london_stop_time.hour - header.utc_datetime_stop.hour == 0

    assert berlin_start_time.hour - london_start_time.hour == 1
    assert berlin_stop_time.hour - london_stop_time.hour == 1


@pytest.mark.parametrize("attr", ("local_datetime_start", "local_datetime_stop"))
def test_timezone_error(dataset_master_simple, attr):
    header = dataset_master_simple[0].info

    header.timezone = None
    with pytest.raises(ValueError) as e:
        getattr(header, attr)

    assert "timezone" in str(e)
