# Calibration

Calibration of IMUs is an important step to obtain accurate measurements.
This library uses the [SensorCalibration](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/sensorcalibration) library to apply calibrations to a Dataset.

## Obtaining a Calibration

### Factory Calibration

If no accurate senor readings are required, applying the factory calibration obtained from the datasheet of the IMU sensor might be sufficient.
These calibrations can be directly applied on a dataset without any further work:

```python
ds = ... # Dataset Object
cal_df = ds.factory_calibrate_imu()
``` 

### Reference Calibrations

For many sensors reference calibrations exist (note: these are different from factory calibrations, which are only based on values from the datasheet).
All reference calibrations are stored in the [NilsPodRefCal](https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal) python package.
To gain access to these calibrations install the package following the instructions provided in its README.

**NOTE**: Usually, it is always a good idea to use the reference calibrations.
However, if the last available reference calibration was far before the actual measurements (multiple month/years) or the measurement was performed under abnormal enviromental conditions,
a new custom calibration should be obtained. 

### Custom Calibrations

First it is necessary to perform a calibration measurement.
The exact measurement protocol will depend on the calibration method.
See the [SensorCalibration](https://mad-srv.informatik.uni-erlangen.de/MadLab/GaitAnalysis/sensorcalibration) library for more information on how to perform the different calibration measurements.
Using the same library a calibration measurement can be converted into a `CalibrationInfo` object:

```python
from nilspodlib.dataset import Dataset
from imucal import FerrarisCalibration  # This example shows a ferraris calibration, but any other method will work similar

path = ... # Path to the .bin file of the calibration measurement

ds = Dataset.from_bin_file(path)
data = ds.imu_data_as_df()
cal, _ = FerrarisCalibration.from_interactive_plot(data, ds.info.sampling_rate_hz)
cal_info = cal.compute_calibration_matrix()
```

This Python object can either be used directly to apply a calibration in the context of this library or can be stored in `.json` file (which can also be directly used with this library).
Note, that *NilsPodLib* assumes a certain naming scheme to make it easier to search through a list of `.json` files.
Therefore, it is suggested to use the `save_calibration` function of this library instead of the simple `.to_json` method provided by the *SensorCalibration* library:

```python
from nilspodlib.calibration_utils import save_calibration

save_calibration(cal_info, ds.info.sensor_id, ds.info.utc_datetime_start, '/my/custom/cal/folder')  # This will save a json with the correct nameing scheme in the custom cal folder.
```


## Apply a Calibration 

### Finding a suitable Calibration

The first step is to locate a suitable calibration for the sensor.
The library provides a couple of tools for this, which can be found in `NilsPodLib.calibration_utils`.

The most common once are shown briefly.
Note, that all the functions shown below support an optional `folder` argument.
If provided only the specified folder will be searched.
If not, the **reference calibrations will be used automatically**. 

List all calibrations, which belong to a sensor:

```python
from nilspodlib.calibration_utils import find_calibrations_for_sensor

ds = ...  # Dataset object
cals = find_calibrations_for_sensor(ds.info.sensor_id)
print(cals)  # Will print a list of file path
cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required
```

Find the calibrations, which belong to a sensor and is closest to the measurement date:

```python
from nilspodlib.calibration_utils import find_closest_calibration_to_date

ds = ...  # Dataset object
cal = find_closest_calibration_to_date(ds.info.sensor_id, ds.info.utc_datetime_start)
print(cal)  # Will print the path to a single calibration
```

Filter for calibrations of one type:

This is available for all search functions. For a full list of possible calibration types see the *SensorCalibration* library.

```python
from nilspodlib.calibration_utils import find_calibrations_for_sensor

ds = ...  # Dataset object
cals = find_calibrations_for_sensor(ds.info.sensor_id, filter_cal_type='turntable')
print(cals)  # Will print a list of all turntable calibrations available for the sensor
cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required
```

Search a set of custom calibrations:

This is available for all search functions.

```python
from nilspodlib.calibration_utils import find_calibrations_for_sensor

ds = ...  # Dataset object
cals = find_calibrations_for_sensor(ds.info.sensor_id, folder='/my/custom/cal/folder')
print(cals)  # Will print a list of all calibrations found in the custom calibration folder
cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required
```

Using the OOP interface:

Instead of using the functions provided by the `calibration_utils` module, the same functions can be invekoed as methods on the dataset.

```python
ds = ...  # Dataset object

cals = ds.find_calibrations()
print(cals)  # Will print a list of file path
cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required
```

```python
ds = ...  # Dataset object
cal = ds.find_closest_calibration()
print(cal)  # Will print the path to a single calibration
```

On a session object:

```python
session = ...  # Session object
cals = session.find_closest_calibration()
print(session)  # Will print a list of calibrations, one for each dataset of the session
```

### Performing the Calibration

To apply the calibration the `calibrate_imu` method of the session or the dataset object can be used:

```python
ds = ...  # Dataset object
calibrated_ds = ds.calibrate_imu(cal)
```

```python
session = ...  # Session object
calibrated_session = session.calibrate_imu(cals) # sessions require a list of calibration objects in the same order as the datasets
```
