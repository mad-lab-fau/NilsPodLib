import datetime
import tempfile
from pathlib import Path

import pytest
import numpy as np
from imucal import FerrarisCalibrationInfo

from NilsPodLib.calibration_utils import save_calibration


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

