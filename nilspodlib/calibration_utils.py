"""A set of utilities to save and find calibrations for NilsPod sensors."""

import datetime
import re
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional, Union, Callable

from nilspodlib.utils import path_t

if TYPE_CHECKING:
    from imucal import CalibrationInfo  # noqa: F401


def save_calibration(
    calibration: "CalibrationInfo",
    sensor_id: str,
    cal_time: datetime.datetime,
    folder: path_t,
    folder_structure: str = "",
) -> Path:
    """Save a calibration info object in the correct format and file name for NilsPods.

    The files will be saved in the format `folder/{sensor_id}_%Y-%m-%d_%H-%M.json` by default.
    If you want to recreate the default folder structure of `imucal`, pass
    `folder_structure="{sensor_id}/{cal_info.CAL_TYPE}"` to the function.

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
    folder_structure :
        A valid formatted Python string using the `{}` syntax.
        `sensor_id`, calibration as the name `cal_info` and kwargs will be passed to the `str.format` as keyword
        arguments and can be used in the string.

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
    from imucal.management import save_calibration_info  # noqa: F401

    return save_calibration_info(
        cal_info=calibration, sensor_id=sensor_id, cal_time=cal_time, folder=folder, folder_structure=folder_structure
    )


def find_calibrations_for_sensor(
    sensor_id: str,
    folder: Optional[path_t] = None,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    custom_validator: Optional[Callable[["CalibrationInfo"], bool]] = None,
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
        This will look for the `CalType` inside the json file and could cause performance issues with many calibration
        files.
        If None, all found files will be returned.
        For possible values, see the `imucal` library.
    custom_validator :
        A custom function that will be called with the CalibrationInfo object of each potential match.
        This needs to load the json file of each match and could cause performance issues with many calibration files.
    ignore_file_not_found :
        If True this function will not raise an error, but rather return an empty list, if no
        calibration files were found for the specific sensor_type.

    Returns
    -------
        list_of_cals
            List of paths pointing to available calibration objects.

    """
    if not folder:
        folder = _check_ref_cal_folder()

    from imucal.management import find_calibration_info_for_sensor  # noqa: F401

    return find_calibration_info_for_sensor(
        sensor_id=sensor_id,
        folder=folder,
        recursive=recursive,
        filter_cal_type=filter_cal_type,
        custom_validator=custom_validator,
        ignore_file_not_found=ignore_file_not_found,
    )


def find_closest_calibration_to_date(
    sensor_id: str,
    cal_time: datetime.datetime,
    folder: Optional[path_t] = None,
    recursive: bool = True,
    filter_cal_type: Optional[str] = None,
    custom_validator: Optional[Callable[["CalibrationInfo"], bool]] = None,
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
        This will look for the `CalType` inside the json file and could cause performance issues with many calibration
        files.
        If None, all found files will be returned.
        For possible values, see the `imucal` library.
    custom_validator :
        A custom function that will be called with the CalibrationInfo object of each potential match.
        This needs to load the json file of each match and could cause performance issues with many calibration files.
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
    if not folder:
        folder = _check_ref_cal_folder()

    from imucal.management import find_closest_calibration_info_to_date  # noqa: F401

    return find_closest_calibration_info_to_date(
        sensor_id=sensor_id,
        cal_time=cal_time,
        folder=folder,
        recursive=recursive,
        filter_cal_type=filter_cal_type,
        custom_validator=custom_validator,
        before_after=before_after,
        warn_thres=warn_thres,
        ignore_file_not_found=ignore_file_not_found,
    )


def load_and_check_cal_info(calibration: Union["CalibrationInfo", path_t]) -> "CalibrationInfo":
    """Load a calibration from path or check if the provided object is already a valid calibration."""
    from imucal import CalibrationInfo  # noqa: import-outside-toplevel

    if isinstance(calibration, (Path, str)):
        from imucal.management import load_calibration_info  # noqa: F401

        calibration = load_calibration_info(calibration)
    if not isinstance(calibration, CalibrationInfo):
        raise ValueError("No valid CalibrationInfo object provided")
    return calibration


def _check_ref_cal_folder() -> Path:
    try:
        from NilsPodRefCal import CAL_PATH  # noqa: import-outside-toplevel
    except ImportError:
        raise ImportError(
            "The module NilsPodRefCal is not installed. If you want support for default calibrations, "
            "please install it from "
            "https://mad-srv.informatik.uni-erlangen.de/MadLab/portabilestools/nilspodrefcal."
        )
    return CAL_PATH
