# -*- coding: utf-8 -*-
"""
Single Dataset
==============

A simple example on how to work with a single Dataset.
"""

import matplotlib.pyplot as plt
from pathlib import Path

from nilspodlib import Dataset

FILEPATH = Path("../tests/test_data/synced_sample_session/NilsPodX-7FAD_20190430_0933.bin")

# Create a Dataset Object from the bin file
dataset = Dataset.from_bin_file(FILEPATH)

# You can access the metainformation about your dataset using the `info` attr.
# For a full list of available attributes see nilspodlib.header.HeaderFields
print("Sensor ID:", dataset.info.sensor_id)
print("Start Date (UTC):", dataset.info.utc_datetime_start)
print("Stop Date (UTC):", dataset.info.utc_datetime_stop)
print("Enabled Sensors:", dataset.info.enabled_sensors)

# You can access the individual sensor data directly from the dataset object using the names provided
# in dataset.info.enabled_sensors
datastream_acc = dataset.acc

# If a sensor is disabled, this will return `None`
print("Analog is disabled:", dataset.analog is None)

# Access the data of a datastream object as a numpy.array using the `data` attribute
print("The acc recordings are {}D and have the length {}".format(*datastream_acc.data.T.shape))

# Convenience methods are available for common operations. E.g. Norm or downsample
plt.figure()
plt.title("Acc Norm")
plt.plot(datastream_acc.norm())
plt.show()

downsampled_datastream = datastream_acc.downsample(factor=2)
print("The new datastream has a length of", len(downsampled_datastream.data))

# However, for many operations it makes more sense to apply them to the Dataset instead of the Datastream.
# This will apply the operations to all Datastream and return a new Dataset object

downsampled_dataset = dataset.downsample(factor=2)
print("Acc has now a length of", len(downsampled_dataset.acc.data))
print("Gyro has now a length of", len(downsampled_dataset.gyro.data))

# By default this returns a copy of the dataset and all datastreams. If this is a performance concern, the dataset can
# be modified inplace:

downsampled_dataset = dataset.downsample(factor=2, inplace=True)
print("The old and the new are identical:", id(dataset) == id(downsampled_dataset))

# Usually, before using any data it needs to be calibrated. The dataset object offers factory_calibrations for all
# important sensors. These convert the datastreams into physical units

dataset_cal = dataset.factory_calibrate_baro()
# For acc and gyro a convenience method is provided.
dataset_cal = dataset_cal.factory_calibrate_imu()

# However, for more precise measurements an actual IMU Calibration using the `calibrate_{acc,gyro,imu}` methods should
# be performed.

# After calibration and initial operations on all datastreams, the easiest way to interface with further processing
# pipelines is a conversion into a pandas DataFrame

df = dataset_cal.data_as_df()

print(df.head())

df.plot()
plt.show()
