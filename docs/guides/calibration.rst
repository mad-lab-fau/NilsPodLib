===========
Calibration
===========

.. warning::
    The handling of calibrations has changed in version 3.0..
    If you are using an earlier version, this information does not apply.
    On information on how to update, see the migration guide in the changelog.

Calibration of IMUs is an important step to obtain accurate measurements.
This library uses the
`imucal <https://github.com/mad-lab-fau/imucal>`_ library to apply calibrations to a Dataset.

Obtaining a Calibration
=======================

Factory Calibration
-------------------

All value provided by the nilspodlib are automatically factory calibrated after loading the binary file.
This means that all values have the expected physical units: Accelerometer (m/s^2), Gyroscope (deg/s), Barometer (mbar),
Temperature (C).
For many applications, this is already sufficient.
However, in particular for the IMU it is preferable to apply a proper calibration to the data to refine the results.
For more information on this you can check the following sections.

Reference Calibrations
----------------------

For many sensors reference calibrations exist (note: these are different from factory calibrations, which are only based
on values from the datasheet).
All reference calibrations are stored in the
`NilsPodRefCal <https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal>`__
python package.
To gain access to these calibrations install the package following the instructions provided in its README.

.. warning::
    `NilsPodRefCal` is a package internal to the FAU MaD-Lab. If you are not a member of the Lab, you can not obtain
    calibration files this way.
    Please contact Portabiles (or whomever provided you NilsPods) for further information and to get potential reference
    calibrations.

.. note::
    Usually, it is always a good idea to use the reference calibrations.
    However, if the last available reference calibration was far before the actual measurements (multiple month/years)
    or the measurement was performed under abnormal environmental conditions, a new custom calibration should be
    obtained.

Custom Calibrations
-------------------

First it is necessary to perform a calibration measurement.
The exact measurement protocol will depend on the calibration method.
See the `imucal <https://github.com/mad-lab-fau/imucal>`__ library for more information on how to perform the different
calibration measurements.

The `imucal` library does not specifically support the nilspodlib, but you can easily use it to create new calibration
files for your sensors.
We recommend (if you do not own professional calibration equipment), to perform a *Ferraris Calibration*.
Follow the instructions provided in `imucal` and record all calibration motions in a single session.
You can then simply load the session and throw the data into `imucal`.
Make sure to specify the correct units when creating the final calibration.

Check this :ref`example <create_calibration>`  for detailed code instructions.

The final `CalibrationInfo` object can either be used directly to apply a calibration in the context of this library or
can be stored in a `.json` file.
For this we recommend to use the tools provided in this library, as they have some nilspod specific tweaks.

>>> from nilspodlib.calibration_utils import save_calibration
>>> save_calibration(cal_info, ds.info.sensor_id, ds.info.utc_datetime_start, '/my/custom/cal/folder')  # This will save a json with the correct nameing scheme in the custom cal folder.

For further information about managing calibration files, check this
`guide <https://imucal.readthedocs.io/en/latest/guides/cal_store.html>`__.

Apply a Calibration
===================

.. warning::
    If you have calibrations created before updating to nilspodlib version 3.0., you will receive an error, when you try
    to load them.
    You need to **recreate** all of these calibration files if you want ot use them with nilspodlib.
    Simply converting them into the new json format will not suffice, as newer version of the nilspodlib expect
    different input units.
    For more information check the Changelog.

Finding a suitable Calibration
------------------------------

The first step is to locate a suitable calibration for the sensor.
The library provides a couple of tools for this, which can be found in `NilsPodLib.calibration_utils`.

The most common once are shown briefly.
Note, that all the functions shown below support an optional `folder` argument.
If provided only the specified folder will be searched.
If not, the **reference calibrations will be used automatically**.

List all calibrations, which belong to a sensor:

>>> from nilspodlib.calibration_utils import find_calibrations_for_sensor
>>> ds = ...  # Dataset object
>>> cals = find_calibrations_for_sensor(ds.info.sensor_id)
>>> print(cals)  # Will print a list of file path
>>> cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required

Find the calibrations, which belong to a sensor and is closest to the measurement date:

>>> from nilspodlib.calibration_utils import find_closest_calibration_to_date
>>> ds = ...  # Dataset object
>>> cal = find_closest_calibration_to_date(ds.info.sensor_id, ds.info.utc_datetime_start)
>>> print(cal)  # Will print the path to a single calibration

Filter for calibrations of one type:

This is available for all search functions.
For a full list of possible calibration types see the *SensorCalibration* library.

>>> from nilspodlib.calibration_utils import find_calibrations_for_sensor
>>> ds = ...  # Dataset object
>>> cals = find_calibrations_for_sensor(ds.info.sensor_id, filter_cal_type='turntable')
>>> print(cals)  # Will print a list of all turntable calibrations available for the sensor
>>> cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required

You can also filter for other information within the exported json file using the `custom_validator` parameter.

Search a set of custom calibrations:

This is available for all search functions.

>>> from nilspodlib.calibration_utils import find_calibrations_for_sensor
>>> ds = ...  # Dataset object
>>> cals = find_calibrations_for_sensor(ds.info.sensor_id, folder='/my/custom/cal/folder')
>>> print(cals)  # Will print a list of all calibrations found in the custom calibration folder
>>> cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required

Using the OOP interface:

Instead of using the functions provided by the `calibration_utils` module, the same functions can be invoked as
methods on the dataset.

>>> ds = ...  # Dataset object
>>> cals = ds.find_calibrations()
>>> print(cals)  # Will print a list of file path
>>> cal = cals[0] # Select the first session. Note the list is not ordered in any way. This means some custom logic for selecting the calibration is required
>>> ds = ...  # Dataset object
>>> cal = ds.find_closest_calibration()
>>> print(cal)  # Will print the path to a single calibration

On a session object:

>>> session = ...  # Session object
>>> cals = session.find_closest_calibration()
>>> print(session)  # Will print a list of calibrations, one for each dataset of the session

Performing the Calibration
--------------------------

To apply the calibration the `calibrate_imu` method of the session or the dataset object can be used:

>>> ds = ...  # Dataset object
>>> calibrated_ds = ds.calibrate_imu(cal)
>>> session = ...  # Session object
>>> calibrated_session = session.calibrate_imu(cals) # sessions require a list of calibration objects in the same order as the datasets
