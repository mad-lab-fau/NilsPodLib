r"""
.. _create_calibration:

Performing a Ferraris calibration
=================================

This example adapts the Ferraris calibration workflow from `imucal
<https://imucal.readthedocs.io/en/latest/guides/ferraris_guide.html>`__ to a
Nilspod recording and keeps the rendered output compact enough for the docs.
"""

from pathlib import Path
import tempfile

import matplotlib.pyplot as plt
import pandas as pd
from imucal import FerrarisCalibration, ferraris_regions_from_section_list
from imucal.management import find_calibration_info_for_sensor
from numpy.linalg import norm

from example_data import EXAMPLE_PATH
from nilspodlib import Dataset
from nilspodlib.calibration_utils import save_calibration


# %%
# Load the recorded calibration session
# -------------------------------------
cal_dataset = Dataset.from_bin_file(
    EXAMPLE_PATH / "example_calibration_recording/example_calibration.bin",
    legacy_support="resolve",
)
data = cal_dataset.imu_data_as_df()

dataset_summary = pd.Series(
    {
        "sensor_id": cal_dataset.info.sensor_id,
        "sampling_rate_hz": cal_dataset.info.sampling_rate_hz,
        "acc_unit": cal_dataset.acc.unit,
        "gyro_unit": cal_dataset.gyro.unit,
        "n_samples": len(data),
    },
    name="value",
).to_frame()

dataset_summary


# %%
# Reuse a saved section annotation
# --------------------------------
# The interactive region picker from :mod:`imucal` is the normal entry point,
# but for a deterministic documentation build we load an annotation from disk.
section_list = pd.read_json(EXAMPLE_PATH / "example_calibration_recording/example_ferraris_session_list.json").T

section_list


# %%
# Compute the calibration
# -----------------------
regions = ferraris_regions_from_section_list(data, section_list)
cal_info = FerrarisCalibration().compute(
    regions,
    sampling_rate_hz=cal_dataset.info.sampling_rate_hz,
    from_acc_unit=cal_dataset.acc.unit,
    from_gyr_unit=cal_dataset.gyro.unit,
    comment="Calibration computed from the bundled Nilspod example recording.",
)

calibration_summary = pd.Series(
    {
        "acc_unit": cal_info.acc_unit,
        "gyro_unit": cal_info.gyr_unit,
        "acc_bias_norm": norm(cal_info.b_a),
        "gyro_bias_norm": norm(cal_info.b_g),
        "comment": cal_info.comment,
    },
    name="value",
).to_frame()

calibration_summary


# %%
# Save the calibration and discover it again
# ------------------------------------------
# In practice you would store calibrations in a long-lived folder. The example
# uses a temporary directory so the gallery run stays self-contained.
with tempfile.TemporaryDirectory() as temp_dir:
    calibration_path = save_calibration(
        cal_info,
        sensor_id=cal_dataset.info.sensor_id,
        cal_time=cal_dataset.info.utc_datetime_start,
        folder=Path(temp_dir),
    )

    discovered_paths = find_calibration_info_for_sensor(cal_dataset.info.sensor_id, Path(temp_dir))
    dataset_paths = cal_dataset.find_calibrations(Path(temp_dir))
    calibrated_dataset = cal_dataset.calibrate_imu(dataset_paths[0])

    calibration_store_summary = pd.DataFrame(
        {
            "source": ["save_calibration", "find_calibration_info_for_sensor", "Dataset.find_calibrations"],
            "result": [
                calibration_path.name,
                discovered_paths[0].name,
                dataset_paths[0].name,
            ],
        }
    ).set_index("source")

calibration_store_summary


# %%
# Compare the raw and calibrated signal
# -------------------------------------
calibrated_data = calibrated_dataset.imu_data_as_df()

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(norm(data.filter(like="acc"), axis=1)[500:1000], label="before calibration", linewidth=1.5)
ax.plot(
    norm(calibrated_data.filter(like="acc"), axis=1)[500:1000],
    label="after calibration",
    linewidth=1.5,
)
ax.set_title("Accelerometer norm before and after calibration")
ax.set_ylabel("norm [m/s^2]")
ax.set_xlabel("sample")
ax.legend()
fig.tight_layout()
