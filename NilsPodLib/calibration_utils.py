import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from NilsPodLib.utils import path_t

if TYPE_CHECKING:
    from imucal import CalibrationInfo


def save_calibration(calibration: 'CalibrationInfo', sensor_id: str, cal_time: datetime.datetime,
                     folder: path_t) -> Path:
    """Saves a calibration info object in the correct format and file name for NilsPods.

    Args:
        calibration: The CalibrationInfo object ot be saved
        sensor_id: The for 4 letter/digit identfier of a sensor, as obtained from
            :py:meth:`NilePodLib.header.Header.sensor_id`
        cal_time: The date and time (min precision) when the calibration was performed
        folder: Basepath of the folder, where the file will be stored.
    """

    f_name = Path(folder) / '{}_{}_{}.json'.format(
        calibration.CAL_TYPE.lower(),
        cal_time.strftime('%Y-%m-%d_%H-%M'),
        sensor_id
    )
    calibration.to_json_file(f_name)
    return f_name
