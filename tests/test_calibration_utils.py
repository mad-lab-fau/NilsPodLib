import copy
import datetime
import tempfile
from pathlib import Path

import pytest
import numpy as np
from imucal import FerrarisCalibrationInfo

from NilsPodLib.calibration_utils import save_calibration, find_calibrations_for_sensor


@pytest.fixture()
def dummy_cal():
    sample_data = {'K_a': np.identity(3),
                   'R_a': np.identity(3),
                   'b_a': np.zeros(3),
                   'K_g': np.identity(3),
                   'R_g': np.identity(3),
                   'K_ga': np.identity(3),
                   'b_g': np.zeros(3)}
    return FerrarisCalibrationInfo(**sample_data)


def test_save_calibration(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        save_calibration(dummy_cal, 'test', datetime.datetime(2000, 10, 3, 13, 22), f)

        assert next(Path(f).glob('*')).name == 'test_2000-10-03_13-22.json'


@pytest.mark.parametrize('sensor_id', [
    '12345',
    'tes*',
    'b da',
    'tes'
])
def test_save_cal_id_validation(dummy_cal, sensor_id):
    with tempfile.TemporaryDirectory() as f:
        with pytest.raises(ValueError):
            save_calibration(dummy_cal, sensor_id, datetime.datetime(2000, 10, 3, 13, 22), f)


@pytest.fixture()
def dummy_cal_folder(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        for sid in ['tes1', 'tes2', 'tes3']:
            for day in range(10, 20):
                date = datetime.datetime(2000, 10, 3, 13, day)
                save_calibration(dummy_cal, sid, date, f)
        yield f

@pytest.fixture()
def dummy_cal_folder_recursive(dummy_cal):
    with tempfile.TemporaryDirectory() as f:
        for s in ['sub1', 'sub2']:
            new_cal = copy.deepcopy(dummy_cal)
            new_cal.CAL_TYPE = s
            for sid in ['tes1', 'tes2', 'tes3']:
                for day in range(10, 20):
                    date = datetime.datetime(2000, 10, 3, 13, day)
                    save_calibration(new_cal, sid, date, Path(f) / s)
        yield f


def test_find_calibration_simple(dummy_cal_folder):
    cals = find_calibrations_for_sensor('tes1', dummy_cal_folder)

    assert len(cals) == 10
    assert all(['tes1' in str(x) for x in cals])


def test_find_calibration_recursive(dummy_cal_folder_recursive):
    cals = find_calibrations_for_sensor('tes1', dummy_cal_folder_recursive)

    assert len(cals) == 0

    cals = find_calibrations_for_sensor('tes1', dummy_cal_folder_recursive, recursive=True)

    assert len(cals) == 20
    assert all(['tes1' in str(x) for x in cals])


def test_find_calibration_type_filer(dummy_cal_folder_recursive):
    cals = find_calibrations_for_sensor('tes1', dummy_cal_folder_recursive, recursive=True, filter_cal_type='sub1')

    assert len(cals) == 10
    assert all(['tes1' in str(x) for x in cals])
