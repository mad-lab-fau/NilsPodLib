"""A set of utilities to save and find calibrations for NilsPod sensors."""

import datetime
import json
import re
import warnings
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union

import numpy as np

from nilspodlib.exceptions import CalibrationWarning
from nilspodlib.utils import path_t

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401


def save_calibration(
    calibration: "CalibrationInfo", sensor_id: str, cal_time: datetime.datetime, folder: path_t
) -> Path:
    """Save a calibration info object in the correct format and file name for NilsPods.

    The files will be saved in the format: `folder / {sensor_id}_%Y-%m-%d_%H-%M.json`

    The naming schema and format is of course just a suggestion, and any structure can be used as long as it can be
    converted back into a CalibrationInfo object.
    However, following the naming convention will allow to use other calibration utils to search for suitable
    calibration files.

    .. note:: If the folder does not exist it will be created.

    Parameters
    ----------
    calibration :
        The CalibrationInfo object ot be saved
    sensor_id :
        The for 4 letter/digit identifier of a sensor_type, as obtained from
        :py:meth:`nilspodlib.header.Header.sensor_id`
    cal_time :
        The date and time (min precision) when the calibration was performed. It is preferable to pass this
        value in UTC timezone, as this is in line with the time handling in the rest of the library.
    folder :
        Basepath of the folder, where the file will be stored.

    Returns
    -------
    output_file_name
        The name under which the calibration file was saved

    """
    if not re.fullmatch(r"\w{4}", sensor_id):
        raise ValueError(
            "The sensor_id is expected to be a 4 symbols string only containing numbers or letters, not {}".format(
                sensor_id
            )
        )
    Path(folder).mkdir(parents=True, exist_ok=True)
    f_name = Path(folder) / "{}_{}.json".format(sensor_id.lower(), cal_time.strftime("%Y-%m-%d_%H-%M"))
    calibration.to_json_file(f_name)
    return f_name


def find_calibrations_for_sensor(
    sensor_id: str,
    folder: Optional[path_t] = None,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    ignore_file_not_found: Optional[bool] = False,
) -> List[Path]:
    """Find possible calibration files based on the filename.

    As this only checks the filenames, this might return false positives depending on your folder structure and naming.

    Parameters
    ----------
    sensor_id :
        The for 4 letter/digit identifier of a sensor_type, as obtained from
        :py:meth:`nilspodlib.header.Header.sensor_id`
    folder :
        Basepath of the folder to search. If None, tries to find a default calibration
    recursive :
        If the folder should be searched recursive or not.
    filter_cal_type :
        Whether only files obtain with a certain calibration type should be found.
        This will look for the `CalType` inside the json file and hence cause performance problems.
        If None, all found files will be returned.
        For possible values, see the `imucal` library.
    ignore_file_not_found :
        If True this function will not raise an error, but rather return an empty list, if no
        calibration files were found for the specific sensor_type.

    Returns
    -------
        list_of_cals
            List of paths pointing to available calibration objects.

    """
    if not folder:
        try:
            from NilsPodRefCal import CAL_PATH  # noqa: import-outside-toplevel
        except ImportError:
            raise ImportError(
                "The module NilsPodRefCal is not installed. If you want support for default calibrations,"
                " please install it from"
                " https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal"
            )
        folder = CAL_PATH

    method = "glob"
    if recursive is True:
        method = "rglob"

    r = sensor_id.lower() + r"_\d{4}-\d{2}-\d{2}_\d{2}-\d{2}"

    matches = [f for f in getattr(Path(folder), method)("{}_*.json".format(sensor_id)) if re.fullmatch(r, f.stem)]

    if filter_cal_type:
        matches = [f for f in matches if json.load(f.open())["cal_type"].lower() == filter_cal_type.lower()]

    if not matches and ignore_file_not_found is not True:
        raise ValueError("No Calibration for the sensor_type with the id {} could be found".format(sensor_id))
    return matches


def find_closest_calibration_to_date(
    sensor_id: str,
    cal_time: datetime.datetime,
    folder: Optional[path_t] = None,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    before_after: Optional[str] = None,
    warn_thres: datetime.timedelta = datetime.timedelta(days=30),  # noqa E252
    ignore_file_not_found: Optional[bool] = False,
) -> Optional[Path]:
    """Find the calibration file for a sensor_type, that is closes to a given date.

    As this only checks the filenames, this might return a false positive depending on your folder structure and naming.

    Parameters
    ----------
    sensor_id :
        The for 4 letter/digit identifier of a sensor_type, as obtained from
        :py:meth:`nilspodlib.header.Header.sensor_id`
    cal_time :
        time and date to look for
    folder :
        Basepath of the folder to search. If None, tries to find a default calibration
    recursive :
        If the folder should be searched recursive or not.
    filter_cal_type :
        Whether only files obtain with a certain calibration type should be found.
        This will look for the `CalType` inside the json file and hence cause performance problems.
        If None, all found files will be returned.
        For possible values, see the `imucal` library.
    before_after :
        Can either be 'before' or 'after', if the search should be limited to calibrations that were
        either before or after the specified date. If None the closest value ignoring if it was before or after the
        measurement.
    warn_thres :
        If the distance to the closest calibration is larger than this threshold, a warning is emitted
    ignore_file_not_found :
        If True this function will not raise an error, but rather return `None`, if no
        calibration files were found for the specific sensor_type.

    Notes
    -----
    If there are multiple calibrations that have the same date/hour/minute distance form the measurement,
    the calibration before the measurement will be chosen. This can be overwritten using the `before_after` para.

    See Also
    --------
    nilspodlib.calibration_utils.find_calibrations_for_sensor

    Returns
    -------
    cal_file_path or None
        The path to a suitable calibration file, or `None`, if no suitable file could be found.

    """
    if before_after not in ("before", "after", None):
        raise ValueError('Invalid value for `before_after`. Only "before", "after" or None are allowed')

    potential_list = find_calibrations_for_sensor(
        sensor_id=sensor_id,
        folder=folder,
        recursive=recursive,
        filter_cal_type=filter_cal_type,
        ignore_file_not_found=ignore_file_not_found,
    )
    if not potential_list:
        if ignore_file_not_found is True:
            return None
        raise ValueError("No Calibration for the sensor_type with the id {} could be found".format(sensor_id))

    dates = [datetime.datetime.strptime("_".join(d.stem.split("_")[1:]), "%Y-%m-%d_%H-%M") for d in potential_list]

    dates = np.array(dates, dtype="datetime64[s]")
    potential_list, _ = zip(*sorted(zip(potential_list, dates), key=lambda x: x[1]))
    dates.sort()

    diffs = (dates - np.datetime64(cal_time, "s")).astype(float)

    if before_after == "after":
        diffs[diffs < 0] = np.nan
    elif before_after == "before":
        diffs[diffs > 0] = np.nan

    if np.all(diffs) == np.nan:
        raise ValueError("No calibrations {} {} were found.".format(before_after, cal_time))

    min_dist = float(np.nanmin(np.abs(diffs)))
    if warn_thres < datetime.timedelta(seconds=min_dist):
        warnings.warn(
            "For the sensor_type {} no calibration could be located that was in {} of the {}."
            "The closest calibration is {} away.".format(
                sensor_id, warn_thres, cal_time, datetime.timedelta(seconds=min_dist)
            ),
            CalibrationWarning,
        )

    return potential_list[int(np.nanargmin(np.abs(diffs)))]


def load_and_check_cal_info(calibration: Union["CalibrationInfo", path_t]) -> "CalibrationInfo":
    """Load a calibration from path or check if the provided object is already a valid calibration."""
    from imucal import CalibrationInfo  # noqa: import-outside-toplevel

    if isinstance(calibration, (Path, str)):
        calibration = CalibrationInfo.from_json_file(calibration)
    if not isinstance(calibration, CalibrationInfo):
        raise ValueError("No valid CalibrationInfo object provided")
    return calibration
