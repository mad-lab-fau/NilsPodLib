import datetime
import json
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional
import re
import numpy as np

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


def find_calibrations_for_sensor(sensor_id: str, folder: path_t, recursive: bool = False,
                                 filter_cal_type: Optional[str] = None) -> List[Path]:
    """Find possible calibration files based on the filename.

    As this only checks the filenames, this might return false positives depending on your folder structure and naming.

    Args:
        sensor_id: The for 4 letter/digit identfier of a sensor, as obtained from
            :py:meth:`NilePodLib.header.Header.sensor_id`
        folder: Basepath of the folder to search
        recursive: If the folder should be searched recursive or not.
        filter_cal_type: Whether only files obtain with a certain calibration type should be found.
            This will look for the `CalType` inside the json file and hence cause performance problems.
            If None, all found files will be returned.
            For possible values, see the `imucal` library.
    """
    # TODO: Test
    method = 'glob'
    if recursive is True:
        method = 'rglob'

    r = r'\d{4}-\d{2}-\d{2}_\d{2}-\d{2}_' + sensor_id.lower()

    potential_matches = [f for f in getattr(Path(folder), method)('*_{}.json'.format(sensor_id)) if
                         re.fullmatch(f.stem, r)]

    if filter_cal_type is None:
        return potential_matches

    return [f for f in potential_matches if json.load(f.open())['cal_type'] == filter_cal_type]


def find_closes_calibration_to_date(sensor_id: str, cal_time: datetime.datetime, folder: path_t,
                                    recursive: bool = False,
                                    filter_cal_type: Optional[str] = None, before_after: Optional[str] = None) -> Path:
    """Find the calibration file for a sensor, that is closes to a given date.

    As this only checks the filenames, this might return false positives depending on your folder structure and naming.

    Args:
        sensor_id: The for 4 letter/digit identfier of a sensor, as obtained from
            :py:meth:`NilePodLib.header.Header.sensor_id`
        cal_time: time and date to look for
        folder: Basepath of the folder to search
        recursive: If the folder should be searched recursive or not.
        filter_cal_type: Whether only files obtain with a certain calibration type should be found.
            This will look for the `CalType` inside the json file and hence cause performance problems.
            If None, all found files will be returned.
            For possible values, see the `imucal` library.
        before_after: Can either be 'before' or 'after', if the search should be limited to calibrations that were
            either before or after the specified date.

    See Also:
        :py:func:`NilsPodLib.calibration_utils.find_calibrations_for_sensor`
    """
    # TODO: Test
    if before_after not in ('before', 'after', None):
        raise ValueError('Invalid value for `before_after`. Only "before", "after" or None are allowed')

    potential_list = find_calibrations_for_sensor(sensor_id=sensor_id, folder=folder, recursive=recursive,
                                                  filter_cal_type=filter_cal_type)
    dates = [datetime.datetime.strptime('_'.join(d.stem.split('_')[:2]), '%Y-%m-%d_%H-%M') for d in potential_list]

    dates = np.array(dates, dtype=np.datetime64)

    diffs = dates - cal_time

    if before_after == 'after':
        diffs[diffs < 0] = np.nan
    elif before_after == 'before':
        diffs[diffs > 0] = np.nan

    if np.all(diffs) == np.nan:
        raise ValueError('No calibrations {} {} were found.'.format(before_after, cal_time))

    return potential_list[int(np.nanargmin(diffs))]
