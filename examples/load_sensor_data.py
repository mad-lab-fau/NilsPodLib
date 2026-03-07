"""
Single Dataset
==============

A simple example on how to work with a single Dataset.
"""

from pathlib import Path

import matplotlib.pyplot as plt

from nilspodlib import Dataset


def _repo_root() -> Path:
    search_roots = [Path.cwd()]
    if "__file__" in globals():
        search_roots.insert(0, Path(__file__).resolve().parent)

    for root in search_roots:
        for parent in (root, *root.parents):
            if (parent / "pyproject.toml").exists():
                return parent
    raise FileNotFoundError("Could not locate the repository root from the example path.")


# %%
# Load the dataset
# ----------------
FILEPATH = _repo_root() / "tests/test_data/synced_sample_session/NilsPodX-7FAD_20190430_0933.bin"

# Create a Dataset Object from the bin file
dataset = Dataset.from_bin_file(FILEPATH)

# You can access the metainformation about your dataset using the `info` attr.
# For a full list of available attributes see nilspodlib.header._HeaderFields
print("Sensor ID:", dataset.info.sensor_id)
print("Start Date (UTC):", dataset.info.utc_datetime_start)
print("Stop Date (UTC):", dataset.info.utc_datetime_stop)
print("Enabled Sensors:", dataset.info.enabled_sensors)


# %%
# Work with individual datastreams
# --------------------------------
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


# %%
# Apply operations to the full dataset
# ------------------------------------
# However, for many operations it makes more sense to apply them to the Dataset instead of the Datastream.
# This will apply the operations to all Datastream and return a new Dataset object

downsampled_dataset = dataset.downsample(factor=2)
print("Acc has now a length of", len(downsampled_dataset.acc.data))
print("Gyro has now a length of", len(downsampled_dataset.gyro.data))

# By default this returns a copy of the dataset and all datastreams. If this is a performance concern, the dataset can
# be modified inplace:

downsampled_dataset = dataset.downsample(factor=2, inplace=True)
print("The old and the new are identical:", id(dataset) == id(downsampled_dataset))

# At this point you would usually apply a calibration to the IMU data (see other examples)


# %%
# Export the data as dataframe
# ----------------------------
# After calibration and initial operations on all datastreams, the easiest way to interface with further processing
# pipelines is a conversion into a pandas DataFrame

df = dataset.data_as_df()

print(df.head())

df.plot()
plt.show()
