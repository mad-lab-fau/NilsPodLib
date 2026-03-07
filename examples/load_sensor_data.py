"""
Working with a single recording
===============================

This example loads one binary recording, inspects the available metadata, and
shows a few common dataset operations that render well in the documentation.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
# Load a sample dataset
# ---------------------
# The gallery executes examples from a generated location, so the input file is
# resolved by walking up to the repository root first.
FILEPATH = _repo_root() / "tests/test_data/synced_sample_session/NilsPodX-7FAD_20190430_0933.bin"

dataset = Dataset.from_bin_file(FILEPATH)


# %%
# Inspect the recording metadata
# ------------------------------
sensor_summary = pd.Series(
    {
        "sensor_id": dataset.info.sensor_id,
        "start_utc": dataset.info.utc_datetime_start,
        "stop_utc": dataset.info.utc_datetime_stop,
        "sampling_rate_hz": dataset.info.sampling_rate_hz,
        "enabled_sensors": ", ".join(dataset.info.enabled_sensors),
    },
    name="value",
).to_frame()

sensor_summary


# %%
# Work with individual datastreams
# --------------------------------
acc_stream = dataset.acc

stream_summary = pd.Series(
    {
        "accelerometer_unit": acc_stream.unit,
        "dimensions": acc_stream.data.shape[1],
        "n_samples": len(acc_stream.data),
        "analog_available": dataset.analog is not None,
    },
    name="value",
).to_frame()

stream_summary


# %%
# Visualize a common transformation
# ---------------------------------
# Convenience methods such as :meth:`~nilspodlib.datastream.Datastream.norm` and
# :meth:`~nilspodlib.datastream.Datastream.downsample` are available directly on
# each datastream.
downsampled_acc = acc_stream.downsample(factor=4)
sampling_rate_hz = dataset.info.sampling_rate_hz

time_seconds = pd.RangeIndex(len(acc_stream.data)) / sampling_rate_hz
downsampled_time_seconds = pd.RangeIndex(len(downsampled_acc.data)) / (sampling_rate_hz / 4)

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(time_seconds[:1200], acc_stream.norm()[:1200], label="original", linewidth=1.5)
ax.plot(
    downsampled_time_seconds[:300],
    downsampled_acc.norm()[:300],
    "o-",
    label="downsampled x4",
    markersize=2.5,
)
ax.set_title("Accelerometer norm for the first 10 seconds")
ax.set_xlabel("time [s]")
ax.set_ylabel(f"norm [{acc_stream.unit}]")
ax.legend()
fig.tight_layout()


# %%
# Apply an operation to the full dataset
# --------------------------------------
# Dataset-level helpers apply the same operation to every available datastream.
downsampled_dataset = dataset.downsample(factor=2)

length_comparison = pd.DataFrame(
    {
        "original_samples": {name: len(stream.data) for name, stream in dataset.datastreams},
        "downsampled_samples": {name: len(stream.data) for name, stream in downsampled_dataset.datastreams},
    }
)

length_comparison


# %%
# Export the data for downstream analysis
# ---------------------------------------
# A pandas dataframe is often the easiest hand-off into later processing steps.
dataset.data_as_df().head()
