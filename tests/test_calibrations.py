import numpy as np
import pytest

from imucal import FerrarisCalibrationInfo
from nilspodlib.dataset import Dataset
from nilspodlib.exceptions import RepeatedCalibrationError
from nilspodlib.header import Header

factory_calibrate_sensors = [
    ("acc", 1.0 / 2048),
    ("gyro", 1.0 / 16.384),
    ("baro", 1013.26),
    ("temperature", 23 + 1 / (2 ** 9)),
]

cal_methods = [(("acc",), "calibrate_acc"), (("gyro",), "calibrate_gyro"), (("acc", "gyro"), "calibrate_imu")]


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
    return FerrarisCalibrationInfo(**expected)


@pytest.fixture()
def simple_header():
    return Header(sampling_rate_hz=102.4, acc_range_g=16, gyro_range_dps=2000)


@pytest.mark.parametrize("sensor,calval", factory_calibrate_sensors)
def test_factory_cal(simple_header, sensor, calval):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, "factory_calibrate_" + sensor)()
    assert np.all(getattr(cal_ds, sensor).data == calval)
    assert getattr(cal_ds, sensor).is_calibrated is True


@pytest.mark.parametrize("sensor", list(zip(*factory_calibrate_sensors))[0])
def test_repeated_cal_error_factory_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, "factory_calibrate_" + sensor)()

    with pytest.raises(RepeatedCalibrationError) as e:
        getattr(cal_ds, "factory_calibrate_" + sensor)()
    assert sensor in str(e.value)


@pytest.mark.parametrize("sensor", list(zip(*factory_calibrate_sensors))[0])
def test_non_existent_warning_factory_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({}, np.arange(100), simple_header)
    with pytest.warns(UserWarning) as warn:
        getattr(dataset, "factory_calibrate_" + sensor)()

    assert len(warn) == 1
    assert sensor in warn[0].message.args[0]
    assert "calibration" in warn[0].message.args[0]


@pytest.mark.parametrize("sensor", list(zip(*factory_calibrate_sensors))[0])
def test_inplace_factor_cal(simple_header, sensor):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = getattr(dataset, "factory_calibrate_" + sensor)()
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, "factory_calibrate_" + sensor)(inplace=True)
    assert id(cal_ds) == id(dataset)
    assert id(getattr(cal_ds, sensor)) == id(getattr(dataset, sensor))

    dataset = Dataset({sensor: np.ones(100)}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, "factory_calibrate_" + sensor)(inplace=False)
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))


def test_imu_cal(simple_header, simple_calibration):
    simple_header.enabled_sensors = ("acc", "gyro")

    dataset = Dataset({"acc": np.ones((100, 3)), "gyro": np.ones((100, 3))}, np.arange(100), simple_header)
    cal_ds = dataset.calibrate_imu(simple_calibration)
    assert np.all(cal_ds.gyro.data == 1.0 / 3)
    assert np.all(cal_ds.acc.data == 1.0 / 2)
    assert cal_ds.gyro.is_calibrated is True
    assert cal_ds.acc.is_calibrated is True


@pytest.mark.parametrize("sensor,calval", [("acc", 1.0 / 2), ("gyro", 1.0 / 3)])
def test_acc_gyro_cal(simple_header, simple_calibration, sensor, calval):
    simple_header.enabled_sensors = (sensor,)

    dataset = Dataset({sensor: np.ones((100, 3))}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, "calibrate_" + sensor)(simple_calibration)
    assert np.all(getattr(cal_ds, sensor).data == calval)
    assert getattr(cal_ds, sensor).is_calibrated is True


@pytest.mark.parametrize("sensors, method", cal_methods)
def test_repeated_cal_error(simple_header, simple_calibration, sensors, method):
    simple_header.enabled_sensors = sensors

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, method)(simple_calibration)

    with pytest.raises(RepeatedCalibrationError) as e:
        getattr(cal_ds, method)(simple_calibration)
    assert sensors[0] in str(e.value)


@pytest.mark.parametrize("sensors, method", cal_methods)
def test_non_existent_warning(simple_header, simple_calibration, sensors, method):
    simple_header.enabled_sensors = sensors

    dataset = Dataset({}, np.arange(100), simple_header)
    with pytest.warns(UserWarning) as warn:
        getattr(dataset, method)(simple_calibration)

    if method == "calibrate_imu":
        assert len(warn) == 2
    else:
        assert len(warn) == 1
    for m in warn:
        assert any(s in str(m) for s in sensors)
        assert "calibration" in str(m)


@pytest.mark.parametrize("sensors, method", cal_methods)
def test_inplace(simple_header, simple_calibration, sensors, method):
    simple_header.enabled_sensors = sensors

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = getattr(dataset, method)(simple_calibration)
    assert id(cal_ds) != id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, method)(simple_calibration, inplace=True)
    assert id(cal_ds) == id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) == id(getattr(dataset, sensor))

    dataset = Dataset({k: np.ones((100, 3)) for k in sensors}, np.arange(100), simple_header)
    cal_ds = getattr(dataset, method)(simple_calibration, inplace=False)
    assert id(cal_ds) != id(dataset)
    for sensor in sensors:
        assert id(getattr(cal_ds, sensor)) != id(getattr(dataset, sensor))


def test_imu_factory_cal(simple_header):
    simple_header.enabled_sensors = ("acc", "gyro")

    dataset = Dataset({"acc": np.ones(100), "gyro": np.ones(100) * 2}, np.arange(100), simple_header)

    cal_ds = dataset.factory_calibrate_imu()

    assert np.all(cal_ds.acc.data == 1.0 / 2048)
    assert np.all(cal_ds.gyro.data == 2.0 / 16.384)
    assert cal_ds.acc.is_calibrated is True
    assert cal_ds.acc.is_calibrated is True


def test_inplace_imu_factory_cal(simple_header):
    simple_header.enabled_sensors = ("acc", "gyro")

    dataset = Dataset({"acc": np.ones(100), "gyro": np.ones(100) * 2}, np.arange(100), simple_header)
    # default: inplace = False
    cal_ds = dataset.factory_calibrate_imu()
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, "acc")) != id(getattr(dataset, "acc"))
    assert id(getattr(cal_ds, "gyro")) != id(getattr(dataset, "gyro"))

    dataset = Dataset({"acc": np.ones(100), "gyro": np.ones(100) * 2}, np.arange(100), simple_header)
    cal_ds = dataset.factory_calibrate_imu(inplace=True)
    assert id(cal_ds) == id(dataset)
    assert id(getattr(cal_ds, "acc")) == id(getattr(dataset, "acc"))
    assert id(getattr(cal_ds, "gyro")) == id(getattr(dataset, "gyro"))

    dataset = Dataset({"acc": np.ones(100), "gyro": np.ones(100) * 2}, np.arange(100), simple_header)
    cal_ds = dataset.factory_calibrate_imu(inplace=False)
    assert id(cal_ds) != id(dataset)
    assert id(getattr(cal_ds, "acc")) != id(getattr(dataset, "acc"))
    assert id(getattr(cal_ds, "gyro")) != id(getattr(dataset, "gyro"))
