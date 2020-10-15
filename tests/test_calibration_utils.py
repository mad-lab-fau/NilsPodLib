import copy
import datetime
import tempfile
from pathlib import Path

import pytest
import numpy as np

from nilspodlib.exceptions import CalibrationWarning
from imucal import FerrarisCalibrationInfo

from nilspodlib.calibration_utils import (
    save_calibration,
    find_calibrations_for_sensor,
    find_closest_calibration_to_date,
)


@pytest.fixture()
def dummy_cal():
    sample_data = {
        "K_a": np.identity(3),
        "R_a": np.identity(3),
        "b_a": np.zeros(3),
        "K_g": np.identity(3),
        "R_g": np.identity(3),
        "K_ga": np.identity(3),
        "b_g": np.zeros(3),
    }
    return FerrarisCalibrationInfo(**sample_data)


def test_save_calibration(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        save_calibration(dummy_cal, "test", datetime.datetime(2000, 10, 3, 13, 22), f)

        assert next(Path(f).glob("*")).name == "test_2000-10-03_13-22.json"


@pytest.mark.parametrize("sensor_id", ["12345", "tes*", "b da", "tes"])
def test_save_cal_id_validation(dummy_cal, sensor_id):
    with tempfile.TemporaryDirectory() as f:
        with pytest.raises(ValueError):
            save_calibration(dummy_cal, sensor_id, datetime.datetime(2000, 10, 3, 13, 22), f)


@pytest.fixture()
def dummy_cal_folder(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        for sid in ["tes1", "tes2", "tes3"]:
            for min in range(10, 30, 2):
                date = datetime.datetime(2000, 10, 3, 13, min)
                save_calibration(dummy_cal, sid, date, f)
        yield f


@pytest.fixture()
def dummy_cal_folder_recursive(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        for s in ["sub1", "sub2", "Sub3"]:
            new_cal = copy.deepcopy(dummy_cal)
            new_cal.CAL_TYPE = s
            for sid in ["tes1", "tes2", "tes3"]:
                for min in range(10, 30, 2):
                    date = datetime.datetime(2000, 10, 3, 13, min)
                    save_calibration(new_cal, sid, date, Path(f) / s)
        yield f


def test_find_calibration_simple(dummy_cal_folder):
    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder)

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])


def test_find_calibration_non_existent(dummy_cal_folder):
    with pytest.raises(ValueError):
        find_calibrations_for_sensor("wrong_sensor", dummy_cal_folder)

    cals = find_calibrations_for_sensor("wrong_sensor", dummy_cal_folder, ignore_file_not_found=True)

    assert len(cals) == 0


def test_find_calibration_recursive(dummy_cal_folder_recursive):
    with pytest.raises(ValueError):
        find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=False)

    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True)

    assert len(cals) == 30
    assert all(["tes1" in str(x) for x in cals])


def test_find_calibration_type_filter(dummy_cal_folder_recursive):
    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="sub1")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])


def test_find_calibration_type_filter_case_sensitive(dummy_cal_folder_recursive):
    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="Sub1")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])

    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="Sub1")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])

    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="sub1")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])


def test_find_calibration_type_filter_case_sensitive_2(dummy_cal_folder_recursive):
    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="Sub3")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])

    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="Sub3")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])

    cals = find_calibrations_for_sensor("tes1", dummy_cal_folder_recursive, recursive=True, filter_cal_type="SUB3")

    assert len(cals) == 10
    assert all(["tes1" in str(x) for x in cals])


def test_find_closest(dummy_cal_folder):
    cal = find_closest_calibration_to_date("tes1", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder)

    assert cal.name == "tes1_2000-10-03_13-14.json"

    # Test that before and after still return the correct one if there is an exact match
    cal = find_closest_calibration_to_date(
        "tes1", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder, before_after="before"
    )

    assert cal.name == "tes1_2000-10-03_13-14.json"

    cal = find_closest_calibration_to_date(
        "tes1", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder, before_after="after"
    )

    assert cal.name == "tes1_2000-10-03_13-14.json"


def test_find_closest_non_existend(dummy_cal_folder):
    with pytest.raises(ValueError):
        find_closest_calibration_to_date("wrong_sensor", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder)

    cal = find_closest_calibration_to_date(
        "wrong_sensor", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder, ignore_file_not_found=True
    )

    assert cal is None


def test_find_closest_before_after(dummy_cal_folder):
    # Default to earlier if same distance before and after.
    cal = find_closest_calibration_to_date("tes1", datetime.datetime(2000, 10, 3, 13, 15), dummy_cal_folder)

    assert cal.name == "tes1_2000-10-03_13-14.json"

    # Return later if after.
    cal = find_closest_calibration_to_date(
        "tes1", datetime.datetime(2000, 10, 3, 13, 15), dummy_cal_folder, before_after="after"
    )

    assert cal.name == "tes1_2000-10-03_13-16.json"

    # Return later if before.
    cal = find_closest_calibration_to_date(
        "tes1", datetime.datetime(2000, 10, 3, 13, 15), dummy_cal_folder, before_after="before"
    )

    assert cal.name == "tes1_2000-10-03_13-14.json"


def test_find_closest_warning(dummy_cal_folder):
    with pytest.warns(CalibrationWarning) as rec:
        find_closest_calibration_to_date(
            "tes1", datetime.datetime(2000, 10, 3, 13, 15), dummy_cal_folder, warn_thres=datetime.timedelta(seconds=30)
        )

    assert len(rec) == 1

    with pytest.warns(None) as rec:
        find_closest_calibration_to_date(
            "tes1", datetime.datetime(2000, 10, 3, 13, 14), dummy_cal_folder, warn_thres=datetime.timedelta(seconds=30)
        )

    assert len(rec) == 0

    # test default
    with pytest.warns(None) as rec:
        find_closest_calibration_to_date("tes1", datetime.datetime(2000, 10, 3, 13, 15), dummy_cal_folder)

    assert len(rec) == 0


def test_find_default_cal():
    cals = find_calibrations_for_sensor("3d73")

    assert len(cals) > 0


def test_find_default_cal_wrong():
    with pytest.raises(ValueError):
        find_calibrations_for_sensor("FFFF")
