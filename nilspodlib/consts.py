"""Basic constants/names used throughout the lib."""
import numpy as np

#: Byte information of each sensor_type in one sample.
#:
#: Format: Overall number of bytes, number of channels, datatype
SENSOR_SAMPLE_LENGTH = {
    "acc": (6, 3, np.int16),
    "gyro": (6, 3, np.int16),
    "mag": (6, 3, np.int16),
    "baro": (2, 1, np.int16),
    "analog": (3, 3, np.uint8),
    "ecg": (4, 1, np.int32),
    "ppg": (4, 1, np.int32),
    "temperature": (2, 1, np.int16),
    "counter": (4, 1, float),
}

#: Default legends for all sensors
SENSOR_LEGENDS = {
    "acc": tuple("acc_" + x for x in "xyz"),
    "gyro": tuple("gyr_" + x for x in "xyz"),
    "baro": tuple(["baro"]),
    "analog": tuple("analog_" + str(x) for x in range(3)),
    "ecg": tuple(["ecg"]),
    "ppg": tuple(["ppg"]),
    "temperature": tuple(["temp"]),
}

#: Default units for all sensors
SENSOR_UNITS = {"acc": "g", "gyro": "dps", "baro": "mbar", "temperature": "C"}

#: Available sensor_type positions
SENSOR_POS = {0: "undefined", 1: "left foot", 2: "right foot", 3: "hip", 4: "left wrist", 5: "right wrist", 6: "chest"}
