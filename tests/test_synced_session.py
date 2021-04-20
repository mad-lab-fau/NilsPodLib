import datetime

import pytest

import numpy as np
import pytz

from nilspodlib.exceptions import SynchronisationError, SynchronisationWarning
from nilspodlib.session import SyncedSession


@pytest.fixture()
def basic_synced_session(dataset_synced):
    return SyncedSession([v[0] for v in dataset_synced.values()])


def test_basic_init(dataset_synced):
    session = SyncedSession([dataset_synced["master"][0], dataset_synced["slave1"][0]])
    assert session.datasets == tuple([dataset_synced["master"][0], dataset_synced["slave1"][0]])


def test_validate_sync_channel(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    setattr(ds1.info, "sync_channel", getattr(ds2.info, "sync_channel") + 1)
    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert "sync_group" in str(e.value)

    ds1.info.sync_address = ds2.info.sync_address + "a"
    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert "sync_group" in str(e.value)


def test_start_end_validation(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    ds1.info.utc_start = ds2.info.utc_start - 10000
    ds1.info.utc_stop = ds2.info.utc_stop - 5000

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert "overlapping time period" in str(e.value)


@pytest.mark.parametrize(
    "roles, err",
    [
        (("master", "master", "slave"), "exactly 1 master"),
        (("slave", "slave", "slave"), "exactly 1 master"),
        (("master", "slave", "disabled"), "either slave or master"),
    ],
)
def test_two_master_validation(dataset_synced, roles, err):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]
    ds3 = dataset_synced["slave2"][0]

    ds1.info.sync_role = roles[0]
    ds2.info.sync_role = roles[1]
    ds3.info.sync_role = roles[2]

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2, ds3])

    assert err in str(e.value)


def test_validate_sampling_rate(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    ds1.info.sampling_rate_hz = ds2.info.sampling_rate_hz + 5

    with pytest.raises(ValueError) as e:
        SyncedSession([ds1, ds2])

    assert "same sampling rate" in str(e.value)


def test_disable_validation(dataset_synced):
    ds1 = dataset_synced["master"][0]
    ds2 = dataset_synced["slave1"][0]

    ds1.info.sync_address = ds2.info.sync_address + "a"
    with pytest.raises(ValueError):
        SyncedSession([ds1, ds2])

    SyncedSession.VALIDATE_ON_INIT = False
    SyncedSession([ds1, ds2])


def test_master(basic_synced_session, dataset_synced):
    master = dataset_synced["master"][0]
    assert basic_synced_session.master.info.sensor_id == master.info.sensor_id


def test_slaves(basic_synced_session, dataset_synced):
    slaves = basic_synced_session.slaves
    slave_ids = [d.info.sensor_id for d in slaves]
    assert len(slaves) == 2
    assert dataset_synced["slave1"][0].info.sensor_id in slave_ids
    assert dataset_synced["slave2"][0].info.sensor_id in slave_ids


def test_align_to_sync_with_end(basic_synced_session):
    s = basic_synced_session.align_to_syncregion(cut_start=True, cut_end=True, inplace=False)

    start = np.array([d.counter[d.info.sync_index_start] for d in basic_synced_session.slaves]).max()
    stop = np.array([d.counter[d.info.sync_index_stop] for d in basic_synced_session.slaves]).min()
    length = stop - start + 1

    for d in s.datasets:
        assert len(d.counter) == len(d.acc.data)
        assert d.counter[-1] == stop
        assert len(d.counter) == length
        assert np.array_equal(d.counter, s.master.counter)


def test_align_to_sync_without_start(basic_synced_session):
    s = basic_synced_session.align_to_syncregion(cut_start=False, cut_end=False, inplace=False)

    start = np.array(
        [d.counter[d.info.sync_index_start] - d.info.sync_index_start for d in basic_synced_session.datasets]
    ).max()
    stop = np.array(
        [
            d.counter[d.info.sync_index_stop] + len(d.counter) - d.info.sync_index_stop
            for d in basic_synced_session.datasets
        ]
    ).min()
    length = stop - start + 1

    for d in s.datasets:
        assert len(d.counter) == len(d.acc.data)
        assert d.counter[0] == start
        assert d.counter[-1] == stop
        assert len(d.counter) == length
        assert np.array_equal(d.counter, s.master.counter)


def test_align_to_sync_without_end(basic_synced_session):
    s = basic_synced_session
    min_len = np.min([len(d.counter) for d in s.slaves])
    s.slaves[0].cut(stop=min_len - 200, inplace=True)

    s = basic_synced_session.align_to_syncregion(cut_start=True, cut_end=False, inplace=False)

    start = np.array([d.counter[d.info.sync_index_start] for d in basic_synced_session.slaves]).max()
    length = np.array([len(d.counter) - d.info.sync_index_start for d in basic_synced_session.slaves]).min()

    for d in s.datasets:
        assert len(d.counter) == len(d.acc.data)
        assert d.counter[0] == start
        assert len(d.counter) == length
        assert np.array_equal(d.counter, s.master.counter)


def test_align_to_sync_slave_longer_than_master(basic_synced_session):
    s = basic_synced_session
    min_len = np.min([len(d.counter) for d in s.slaves])
    s.master.cut(stop=min_len - 200, inplace=True)

    s = s.align_to_syncregion(cut_start=True, inplace=False)

    for d in s.datasets:
        assert len(d.counter) == len(d.acc.data)
        assert np.array_equal(d.counter, s.master.counter)


def test_align_to_sync_warn_end(basic_synced_session):
    s = basic_synced_session

    with pytest.warns(None) as rec:
        s.align_to_syncregion(cut_start=True, cut_end=False)

    assert len(rec) == 0

    thres = 0
    with pytest.warns(SynchronisationWarning) as rec:
        s.align_to_syncregion(cut_start=True, cut_end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([d.info.sensor_id for d in s.slaves]) in str(rec[0])

    thres = 30
    s.slaves[0].info.sync_index_stop -= int(30 * s.slaves[0].info.sampling_rate_hz)
    with pytest.warns(SynchronisationWarning) as rec:
        s.align_to_syncregion(cut_start=True, cut_end=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([s.slaves[0].info.sensor_id]) in str(rec[0])

    with pytest.warns(None) as rec:
        s.align_to_syncregion(cut_start=True, cut_end=False, warn_thres=None)

    assert len(rec) == 0


def test_align_to_sync_warn_start(basic_synced_session):
    s = basic_synced_session

    with pytest.warns(None) as rec:
        s.align_to_syncregion(cut_end=True, cut_start=False)

    assert len(rec) == 0

    thres = 0
    with pytest.warns(SynchronisationWarning) as rec:
        s.align_to_syncregion(cut_end=True, cut_start=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([d.info.sensor_id for d in s.slaves]) in str(rec[0])

    thres = 30
    s.slaves[0].info.sync_index_start += int(30 * s.slaves[0].info.sampling_rate_hz)
    with pytest.warns(SynchronisationWarning) as rec:
        s.align_to_syncregion(cut_end=True, cut_start=False, warn_thres=thres)

    assert len(rec) == 1
    assert str(thres) in str(rec[0])
    assert str([s.slaves[0].info.sensor_id]) in str(rec[0])

    with pytest.warns(None) as rec:
        s.align_to_syncregion(cut_end=True, cut_start=False, warn_thres=None)

    assert len(rec) == 0


def test_sync_info(dataset_synced):
    dataset, path = dataset_synced["slave1"]
    master = dataset_synced["master"][0]

    info = dataset.info
    assert info.sync_address == master.info.sync_address
    assert info.sync_channel == master.info.sync_channel
    assert info.sync_distance_ms == master.info.sync_distance_ms
    assert info.is_synchronised is True
    assert info.sync_role == "slave"
    assert info.sync_index_stop != 0
    assert info.sync_index_start != 0


@pytest.mark.parametrize(
    "method",
    [
        "session_utc_start",
        "session_utc_stop",
        "session_duration",
        "session_utc_datetime_start",
        "session_utc_datetime_stop",
        "session_local_datetime_start",
        "session_local_datetime_stop",
    ],
)
def test_synced_time_info_error(basic_synced_session, method):
    with pytest.raises(SynchronisationError):
        getattr(basic_synced_session, method)


@pytest.mark.parametrize(
    "method",
    [
        "session_local_datetime_start",
        "session_local_datetime_stop",
    ],
)
def test_synced_timezone_error(basic_synced_session, method):
    basic_synced_session = basic_synced_session.align_to_syncregion(cut_start=True)
    basic_synced_session.master.info.timezone = None
    with pytest.raises(ValueError):
        getattr(basic_synced_session, method)


@pytest.mark.parametrize("timezone", ("Europe/Berlin", "Europe/London"))
def test_session_local_datetime(basic_synced_session, timezone):
    basic_synced_session = basic_synced_session.align_to_syncregion(cut_start=True)
    basic_synced_session.master.info.timezone = timezone

    start = basic_synced_session.session_local_datetime_start
    end = basic_synced_session.session_local_datetime_stop

    assert start == datetime.datetime(2019, 4, 30, 7, 33, 12, 11719, tzinfo=pytz.utc).astimezone(
        pytz.timezone(timezone)
    )
    assert end == datetime.datetime(2019, 4, 30, 7, 33, 58, 857422, tzinfo=pytz.utc).astimezone(pytz.timezone(timezone))


def test_session_utc_datetime(basic_synced_session):
    basic_synced_session = basic_synced_session.align_to_syncregion(cut_start=True)

    start = basic_synced_session.session_utc_datetime_start
    end = basic_synced_session.session_utc_datetime_stop

    assert start == datetime.datetime(2019, 4, 30, 7, 33, 12, 11719, tzinfo=datetime.timezone.utc)
    assert end == datetime.datetime(2019, 4, 30, 7, 33, 58, 857422, tzinfo=datetime.timezone.utc)

    start = basic_synced_session.session_utc_start
    end = basic_synced_session.session_utc_stop

    assert start == 1556609592.011719
    assert end == 1556609638.857422

    duration = basic_synced_session.session_duration

    assert duration == 46.845703125


@pytest.mark.parametrize("start_idx", list(range(0, 2)))
def test_align_sync_region_with_imidiate_sync(basic_synced_session, start_idx):
    """Test edgecases where the sync happens in the first two samples."""
    s = basic_synced_session
    s.slaves[0].info.sync_index_start = start_idx

    s.align_to_syncregion()

    for d in s.datasets:
        assert len(d.counter) == len(d.acc.data)
