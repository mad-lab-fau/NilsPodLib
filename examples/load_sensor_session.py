# -*- coding: utf-8 -*-
"""
Sessions
=========

A simple example showing how to work with Sensor Sessions.
"""
from pathlib import Path
import numpy as np

from nilspodlib import Dataset, Session, SyncedSession

FILEPATH = Path("../tests/test_data/synced_sample_session/")

# A session consists of multiple datasets. By default this is also the way to create one
datasets = [Dataset.from_bin_file(d) for d in FILEPATH.glob("*.bin")]
session = Session(datasets)
print("This session has {} datasets".format(len(session.datasets)))

# However, in many cases it is easier to use one of the Session constructors:
session = Session.from_folder_path(FILEPATH, filter_pattern="*.bin")
print("This session has {} datasets".format(len(session.datasets)))

# Like Datasets contain convenience methods to act on all Datastreams, Sessions provide methods that work on all
# datasets

downsampled_session = session.downsample(factor=2)
for ds in downsampled_session.datasets:
    for name, d in ds.datastreams:
        print("{} of {} has the length {}".format(name, ds.info.sensor_id, len(d.data)))

# Further you can use the Proxy Attribute `info` to access the header infos of all sensors at the same time
print("The included sensors are:", session.info.sensor_id)
print("The samplingrates are:", session.info.sampling_rate_hz)
print("The enabled sensor are:", session.info.enabled_sensors)

# The library differentiates between synchronised and not synchronised session.
# If your session is synchronised your should use a SyncedSession

session = SyncedSession.from_folder_path(FILEPATH)

# This will also validate that all datasets are compatible to be syncronised.
# If you need to switch off this validation, you can disable it using:
SyncedSession.VALIDATE_ON_INIT = False
session = SyncedSession.from_folder_path(FILEPATH)

# For synced sessions you can get the datasets of the master and the slaves separately

print("The master of the session is", session.master.info.sensor_id)
print("The slaves of the session are", [d.info.sensor_id for d in session.slaves])

# To make use of the sync information, all datasets need to be aligned. This can be done using the `cut_to_syncregion`
# method.

cut_session = session.cut_to_syncregion()

# After this all session are aligned and the dataset counter are identical

for d in cut_session.slaves:
    if np.array_equal(d.counter, cut_session.master.counter) is True:
        print("{} has the same counter than master ({})".format(d.info.sensor_id, cut_session.master.info.sensor_id))
