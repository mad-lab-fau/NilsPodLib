r"""
.. _create_calibration:

Performing a Ferraris Calibration with a Nilspod
================================================

This examplke is adapted based on the generic example that can be found in the `imucal` library.

We assume that you recorded a Ferraris session with you IMU unit following our `tutorial
<https://imucal.readthedocs.io/en/latest/guides/ferraris_guide.html>`__ and performed the entire calibration in a single
recording.
"""

# %%
# Loading the session
# -------------------
# First we need to load the recorded session using the nilspodlib.
# We will directly export the data as dataframe that can be used in imucal.
from pathlib import Path

from example_data import EXAMPLE_PATH
from nilspodlib import Dataset

cal_dataset = Dataset.from_bin_file(
    EXAMPLE_PATH / "example_calibration_recording/example_calibration.bin", legacy_support="resolve"
)
data = cal_dataset.imu_data_as_df()

# %%
# Annotating the data
# -------------------
# Now we need to annotate the different sections of the ferraris calibration in the interactive GUI.
# Note, that this will only work, if your Python process runs on your machine and not some kind of sever.
# Otherwise, the GUI will not open.
#
# To start the annotation, run:
#
#   >>> from imucal import ferraris_regions_from_interactive_plot
#   >>> regions, section_list = ferraris_regions_from_interactive_plot(data)
#
# Check the example in `imucal` for full instructions on how to use the GUI.
#
# Instead of performing the annotation in this example, we will load the section list from a previous annotation of the
# data.
# In general it is advisable to save the annotated sections, so that you can rerun the calibration in the future.

import pandas as pd
from imucal import ferraris_regions_from_section_list

section_list = pd.read_json(EXAMPLE_PATH / "example_calibration_recording/example_ferraris_session_list.json").T

section_list

# %%
# This section list can then be used to recreate the regions
regions = ferraris_regions_from_section_list(data, section_list)
regions

# %%
# Now we can calculate the actual calibration parameters.
# For this we will create a instance of `FerrarisCalibration` with the desired settings and then call `compute` with
# the regions we have extracted.
#
# Note that we need to specify the units of the input data.
# We can do that using the information provided in our loaded dataset.
from imucal import FerrarisCalibration

cal = FerrarisCalibration()
cal_info = cal.compute(
    regions,
    sampling_rate_hz=cal_dataset.info.sampling_rate_hz,
    from_acc_unit=cal_dataset.acc.unit,
    from_gyr_unit=cal_dataset.gyro.unit,
    comment="This is a calibration of a Nilspod.",
)


print(cal_info.to_json())

import tempfile

# %%
# The final `cal_info` now contains all information to calibrate future data recordings from the same sensor.
# For now we will save it to disk and then see how to load it again.
# We can use the `calibration_utils` to save the file in a folder structure of our liking (by default just a simple
# folder with special filenames).
#
# Note, that we will use a temporary folder here.
# In reality you would chose some folder where you can keep the calibration files save until eternity.
from nilspodlib.calibration_utils import save_calibration

d = tempfile.TemporaryDirectory()

file_path = save_calibration(
    cal_info, sensor_id=cal_dataset.info.sensor_id, cal_time=cal_dataset.info.utc_datetime_start, folder=Path(d.name)
)
file_path

# %%
# At a later date, we can use the helper functions :func:`~nilspodlib.calibration_utils.find_calibrations_for_sensor`
# and :func:`~nilspodlib.calibration_utils.find_closest_calibration_to_date` to find the calibration again.
from imucal.management import find_calibration_info_for_sensor

cals = find_calibration_info_for_sensor(cal_dataset.info.sensor_id, Path(d.name))
cals

# %%
# Alternatively, we can use the object oriented interface on the loaded dataset, we want to apply the calibration to.
# in this case we will just use the calibration session we have already loaded.

cals = cal_dataset.find_calibrations(Path(d.name))
cals

# %%
# After finding the calibration file, we will apply it to a "new" recording (again, we will just use the calibration
# session as example here).

calibrated_dataset = cal_dataset.calibrate_imu(cals[0])

# %%
# We can see the effect of the calibration, when we plot the acc norm in the beginning of the recording.
# The calibrated values are now much closer to 9.81 m/s^2 compared to before the calibration.
import matplotlib.pyplot as plt
from numpy.linalg import norm

calibrated_data = calibrated_dataset.imu_data_as_df()

plt.figure()
plt.plot(norm(data.filter(like="acc"), axis=1)[500:1000], label="before cal")
plt.plot(norm(calibrated_data.filter(like="acc")[500:1000], axis=1), label="after cal")
plt.legend()
plt.ylabel("acc norm [m/s^2]")
plt.show()

# %%
# Finally, remove temp directory.
d.cleanup()
