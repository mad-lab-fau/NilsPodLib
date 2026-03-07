"""
Working with sensor sessions
============================

This example shows how to load several recordings together, apply operations to
every dataset in the session, and inspect the synchronized session view.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from nilspodlib import Session, SyncedSession


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
# Load all recordings in a folder
# -------------------------------
DATA_DIR = _repo_root() / "tests/test_data/synced_sample_session"

session = Session.from_folder_path(DATA_DIR, filter_pattern="*.bin")

session_overview = pd.DataFrame(
    [
        {
            "sensor_id": dataset.info.sensor_id,
            "sampling_rate_hz": dataset.info.sampling_rate_hz,
            "enabled_sensors": ", ".join(dataset.info.enabled_sensors),
        }
        for dataset in session.datasets
    ]
).set_index("sensor_id")

session_overview


# %%
# Apply operations to every dataset at once
# -----------------------------------------
downsampled_session = session.downsample(factor=2)

length_comparison = pd.DataFrame(
    [
        {
            "sensor_id": original.info.sensor_id,
            "original_acc_samples": len(original.acc.data),
            "downsampled_acc_samples": len(downsampled.acc.data),
        }
        for original, downsampled in zip(session.datasets, downsampled_session.datasets)
    ]
).set_index("sensor_id")

length_comparison


# %%
# Inspect the synchronized session
# --------------------------------
# If the recordings contain synchronization metadata, use
# :class:`~nilspodlib.session.SyncedSession` to access master and slave devices.
synced_session = SyncedSession.from_folder_path(DATA_DIR)
cut_session = synced_session.cut_to_syncregion()

aligned_datasets = [cut_session.master, *cut_session.slaves]
alignment_overview = pd.DataFrame(
    [
        {
            "sensor_id": dataset.info.sensor_id,
            "role": "master" if dataset is cut_session.master else "slave",
            "counter_start": int(dataset.counter[0]),
            "counter_stop": int(dataset.counter[-1]),
            "n_samples": len(dataset.acc.data),
        }
        for dataset in aligned_datasets
    ]
).set_index("sensor_id")

alignment_overview


# %%
# Visualize the aligned accelerometer norms
# -----------------------------------------
common_window = min(len(dataset.acc.data) for dataset in aligned_datasets)
plot_window = min(common_window, 600)

fig, ax = plt.subplots(figsize=(8, 4))
for dataset in aligned_datasets:
    ax.plot(
        range(plot_window),
        dataset.acc.norm()[:plot_window],
        label=dataset.info.sensor_id,
        linewidth=1.5,
    )

ax.set_title("Accelerometer norm after cutting to the sync region")
ax.set_xlabel("sample")
ax.set_ylabel(f"norm [{cut_session.master.acc.unit}]")
ax.legend(title="sensor_id")
fig.tight_layout()
