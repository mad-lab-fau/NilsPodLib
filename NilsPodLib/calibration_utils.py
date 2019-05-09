import datetime
from pathlib import Path
from typing import TYPE_CHECKING, List
import re

from NilsPodLib.utils import path_t

if TYPE_CHECKING:
    from imucal import CalibrationInfo


def save_calibration(calibration: 'CalibrationInfo', sensor_id: str, cal_time: datetime.datetime,
                     folder: path_t) -> Path:
    """Saves a calibration info object in the correct format and file name for NilsPods.

    The files will be saved in the format:
        folder / %Y-%m-%d_%H-%M_{sensor_id}.json

    The naming schema and format is of course just a suggestion, and any structure can be used as long as it can be
    converted back into a CalibrationInfo object.
    However, following the naming convention will allow to use other calibration utils to search for suitable
    calibration files.

    Args:
        calibration: The CalibrationInfo object ot be saved
        sensor_id: The for 4 letter/digit identfier of a sensor, as obtained from
            :py:meth:`NilePodLib.header.Header.sensor_id`
        cal_time: The date and time (min precision) when the calibration was performed
        folder: Basepath of the folder, where the file will be stored.
    """
    # TODO: Test
    f_name = Path(folder) / '{}_{}.json'.format(
        cal_time.strftime('%Y-%m-%d_%H-%M'),
        sensor_id
    )
    calibration.to_json_file(f_name)
    return f_name


def find_calibrations_for_sensor(sensor_id: str, folder: path_t, recursive=False) -> List[Path]:
    """Find possible calibration files based on the filename.

    As this only checks the filenames, this might return false positives depending on your folder structure and naming.

    Args:
        sensor_id: The for 4 letter/digit identfier of a sensor, as obtained from
            :py:meth:`NilePodLib.header.Header.sensor_id`
        folder: Basepath of the folder to search
        recursive: If the folder should be searched recursive or not.
    """
    # TODO: Test
    method = 'glob'
    if recursive is True:
        method = 'rglob'

    r = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_' + sensor_id.lower()

    return [f for f in getattr(Path(folder), method)('*_{}.json'.format(sensor_id)) if re.fullmatch(f.stem, r)]

