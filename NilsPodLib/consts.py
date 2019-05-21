"""Basic constants/names used throughout the lib."""
import numpy as np

#: Byte information of each sensor in one sample.
#:
#: Format: Overall number of bytes, number of channels, datatype
SENSOR_SAMPLE_LENGTH = {
    'acc': (6, 3, np.int16),
    'gyro': (6, 3, np.int16),
    'mag': (6, 3, np.int16),
    'baro': (2, 1, np.int16),
    'analog': (3, 3, np.uint8),
    'ecg': (None, None, None),  # Needs to be implement
    'ppg': (None, None, None),  # Needs to be implement
    'battery': (1, 1, np.uint8)

}

#: Default legends for all sensors
SENSOR_LEGENDS = {
    'acc': tuple('acc_' + x for x in 'xyz'),
    'gyro': tuple('gyr_' + x for x in 'xyz'),
    'baro': tuple(['baro']),
    'battery': tuple(['battery']),
    'analog': tuple('analog_' + str(x) for x in range(3))
}

#: Default units for all sensors
SENSOR_UNITS = {
    'acc': 'g',
    'gyro': 'dps',
    'baro': 'mbar',
    'battery': 'V'
}

#: Available sensor positions
SENSOR_POS = {
    0: 'undefined',
    1: 'left foot',
    2: 'right foot',
    3: 'hip',
    4: 'left wrist',
    5: 'right wrist',
    6: 'chest'
}
