import pytest
from NilsPodLib.dataset import Dataset
from NilsPodLib.header import Header
import numpy as np
from NilsPodLib.utils import RepeatedCalibrationError

calibrate_sensors = [('acc', 1. / 2048),
                     ('gyro', 1. / 16.384),
                     ('baro', 1013.26),
                     ]


@pytest.fixture()
def simple_header():
    return Header(sampling_rate_hz=102.4, acc_range_g=16, gyro_range_dps=2000)


@pytest.mark.parametrize('sensor,calval', calibrate_sensors)
def test_factory_cal(simple_header, sensor, calval):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, 'factory_calibrate_' + sensor)()
    assert np.all(getattr(cal_ds, sensor).data == calval)
    assert getattr(cal_ds, sensor).is_calibrated is True


@pytest.mark.parametrize('sensor', list(zip(*calibrate_sensors))[0])
def test_repeated_cal_error_factory_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, 'factory_calibrate_' + sensor)()

    with pytest.raises(RepeatedCalibrationError) as e:
        getattr(cal_ds, 'factory_calibrate_' + sensor)()
    assert sensor in str(e.value)


@pytest.mark.parametrize('sensor', list(zip(*calibrate_sensors))[0])
def test_non_existent_warning_factory_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({}, np.arange(100), simple_header)
    with pytest.warns(UserWarning) as warn:
        getattr(dataset, 'factory_calibrate_' + sensor)()

    assert len(warn) == 1
    assert sensor in warn[0].message.args[0]
    assert 'calibration' in warn[0].message.args[0]


@pytest.mark.parametrize('sensor', list(zip(*calibrate_sensors))[0])
def test_inplace_factor_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = getattr(dataset, 'factory_calibrate_' + sensor)()
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, 'factory_calibrate_' + sensor)(inplace=True)
    assert id(cal_ds) == id(dataset)
    assert id(getattr(cal_ds, sensor)) == id(getattr(dataset, sensor))

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, 'factory_calibrate_' + sensor)(inplace=False)
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))


def test_imu_factory_cal(simple_header):
    simple_header.enabled_sensors = ('acc', 'gyro')

    dataset = Dataset({'acc': np.ones(100), 'gyro': np.ones(100) * 2}, np.arange(100), simple_header)

    cal_ds = dataset.factory_calibrate_imu()

    assert np.all(cal_ds.acc.data == 1. / 2048)
    assert np.all(cal_ds.gyro.data == 2. / 16.384)
    assert cal_ds.acc.is_calibrated is True
    assert cal_ds.acc.is_calibrated is True


def test_inplace_imu_factory_cal(simple_header):
    simple_header.enabled_sensors = ('acc', 'gyro')

    dataset = Dataset({'acc': np.ones(100), 'gyro': np.ones(100) * 2}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = dataset.factory_calibrate_imu()
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, 'acc')) != id(getattr(dataset, 'acc'))
    assert id(getattr(cal_ds, 'gyro')) != id(getattr(dataset, 'gyro'))

    dataset = Dataset({'acc': np.ones(100), 'gyro': np.ones(100) * 2}, np.arange(100), simple_header)
    cal_ds = dataset.factory_calibrate_imu(inplace=True)
    assert id(cal_ds) == id(dataset)
    assert id(getattr(cal_ds, 'acc')) == id(getattr(dataset, 'acc'))
    assert id(getattr(cal_ds, 'gyro')) == id(getattr(dataset, 'gyro'))

    dataset = Dataset({'acc': np.ones(100), 'gyro': np.ones(100) * 2}, np.arange(100), simple_header)
    cal_ds = dataset.factory_calibrate_imu(inplace=False)
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, 'acc')) != id(getattr(dataset, 'acc'))
    assert id(getattr(cal_ds, 'gyro')) != id(getattr(dataset, 'gyro'))
