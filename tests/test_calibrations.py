import numpy as np
import pytest

from imucal import FerrarisCalibrationInfo
from nilspodlib.consts import GRAV
from nilspodlib.dataset import Dataset
from nilspodlib.exceptions import RepeatedCalibrationError
from nilspodlib.header import Header

factory_calibrate_sensors = [
    ("acc", 1.0 / 2048 * GRAV),
    ("gyro", 1.0 / 16.384),
    ("baro", 1013.26),
    ("temperature", 23 + 1 / (2 ** 9)),
]
factory_calibrate_sensors_dict = dict(factory_calibrate_sensors)


@pytest.fixture()
def simple_calibration():
    expected = dict()
    expected["K_a"] = np.identity(3) * 2
    expected["R_a"] = np.identity(3)
    expected["b_a"] = np.zeros(3)
    expected["K_g"] = np.identity(3) * 3
    expected["R_g"] = np.identity(3)
    expected["b_g"] = np.zeros(3)
    expected["K_ga"] = np.zeros((3, 3))
    expected["from_acc_unit"] = "m/s^2"
    expected["from_gyr_unit"] = "deg/s"
    return FerrarisCalibrationInfo(**expected)


# TODO: Test errors if wrong calibrations are used.


@pytest.fixture()
def simple_header():
    return Header(sampling_rate_hz=102.4, acc_range_g=16, gyro_range_dps=2000)


@pytest.mark.parametrize("sensor, calval", factory_calibrate_sensors)
def test_factory_cal(simple_header, sensor, calval):
    """Test that all sensors are factory calibrated by default on init."""
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    assert np.all(getattr(dataset, sensor).data == calval)
    assert getattr(dataset, sensor).is_factory_calibrated is True
    assert getattr(dataset, sensor).is_calibrated is False


@pytest.mark.parametrize("sensor", list(zip(*factory_calibrate_sensors))[0])
def test_repeated_cal_error_factory_cal(simple_header, sensor):
    """Test that we can not apply factory calibration twice.

    Note: This should never happen, as factory cal methods are private.
    """
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)

    with pytest.raises(RepeatedCalibrationError) as e:
        getattr(dataset, "_factory_calibrate_" + sensor)(getattr(dataset, sensor))
    assert sensor in str(e.value)
    assert "factory-calibrate" in str(e.value)


def test_imu_cal(simple_header, simple_calibration):
    simple_header.enabled_sensors = ("acc", "gyro")

    dataset = Dataset({"acc": np.ones((100, 3)), "gyro": np.ones((100, 3))}, np.arange(100), simple_header)
    cal_ds = dataset.calibrate_imu(simple_calibration)
    assert np.all(cal_ds.gyro.data == factory_calibrate_sensors_dict["gyro"] / 3)
    assert np.all(cal_ds.acc.data == factory_calibrate_sensors_dict["acc"] / 2)
    assert cal_ds.gyro.is_factory_calibrated is True
    assert cal_ds.acc.is_factory_calibrated is True
    assert cal_ds.gyro.is_calibrated is True
    assert cal_ds.acc.is_calibrated is True


def test_repeated_cal_error(simple_header, simple_calibration):
    sensors = ("acc", "gyro")
    simple_header.enabled_sensors = sensors

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = dataset.calibrate_imu(simple_calibration)

    with pytest.raises(RepeatedCalibrationError) as e:
        cal_ds.calibrate_imu(simple_calibration)
    assert sensors[0] in str(e.value)


def test_non_existent_warning(simple_header, simple_calibration):
    sensors = ("acc", "gyro")
    simple_header.enabled_sensors = sensors

    dataset = Dataset({}, np.arange(100), simple_header)
    with pytest.warns(UserWarning) as warn:
        dataset.calibrate_imu(simple_calibration)

    assert len(warn) == 2
    for m in warn:
        assert any(s in str(m) for s in sensors)
        assert "calibration" in str(m)


def test_inplace(simple_header, simple_calibration):
    sensors = ("acc", "gyro")
    simple_header.enabled_sensors = sensors

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = dataset.calibrate_imu(simple_calibration)
    assert id(cal_ds) != id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = dataset.calibrate_imu(simple_calibration, inplace=True)
    assert id(cal_ds) == id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) == id(getattr(dataset, sensor))

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = dataset.calibrate_imu(simple_calibration, inplace=False)
    assert id(cal_ds) != id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))


def test_imu_factory_cal(
    simple_header,
):
    simple_header.enabled_sensors = ("acc", "gyro")

    dataset = Dataset({"acc": np.ones(100), "gyro": np.ones(100) * 2}, np.arange(100), simple_header)

    assert np.all(dataset.acc.data == factory_calibrate_sensors_dict["acc"])
    assert np.all(dataset.gyro.data == factory_calibrate_sensors_dict["gyro"] * 2)
    assert dataset.acc.is_calibrated is False
    assert dataset.acc.is_calibrated is False
    assert dataset.acc.is_factory_calibrated is True
    assert dataset.acc.is_factory_calibrated is True
