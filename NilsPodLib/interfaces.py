from functools import wraps
from typing import Optional, Iterable, Tuple, Union, Any, TypeVar, TYPE_CHECKING, Type, Sequence

import numpy as np
import pandas as pd

from NilsPodLib import Dataset
from NilsPodLib.utils import path_t, inplace_or_copy

T = TypeVar('T')

if TYPE_CHECKING:
    from NilsPodLib.datastream import Datastream  # noqa: F401
    from imucal import CalibrationInfo  # noqa: F401


class CascadingField:
    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        return tuple(getattr(d, self.name) for d in instance.datasets)


def call_dataset(append_doc=True):
    def wrapped(method):
        @wraps(method)
        def cascading_access(*args, **kwargs):
            session = args[0]
            return_vals = tuple(getattr(d, method.__name__)(*args[1:], **kwargs) for d in session.datasets)

            if all(isinstance(d, Dataset) for d in return_vals):
                inplace = kwargs.get('inplace', False)
                s = inplace_or_copy(session, inplace)
                s.datasets = return_vals
                return s
            return return_vals

        if append_doc:
            if cascading_access.__doc__:
                cascading_access.__doc__ += '\n\n'
            else:
                cascading_access.__doc__ = ''
            cascading_access.__doc__ += getattr(Dataset, method.__name__).__doc__
        return cascading_access
    return wrapped


class MultiDataset:
    path: path_t = CascadingField()
    acc: Tuple[Optional['Datastream']] = CascadingField()
    gyro: Tuple[Optional['Datastream']] = CascadingField()
    mag: Tuple[Optional['Datastream']] = CascadingField()
    baro: Tuple[Optional['Datastream']] = CascadingField()
    analog: Tuple[Optional['Datastream']] = CascadingField()
    ecg: Tuple[Optional['Datastream']] = CascadingField()
    ppg: Tuple[Optional['Datastream']] = CascadingField()
    temperature: Tuple[Optional['Datastream']] = CascadingField()
    counter: Tuple[np.ndarray] = CascadingField()

    size: Tuple[int] = CascadingField()
    datastreams: Tuple[Iterable['Datastream']] = CascadingField()

    ACTIVE_SENSORS: Tuple[Tuple[str]] = CascadingField()

    # Dark magic for metaclass
    _base_keys = None

    @call_dataset()
    def calibrate_imu(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def calibrate_acc(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def calibrate_gyro(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def factory_calibrate_imu(self: Type[T], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def factory_calibrate_gyro(self: Type[T], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def factory_calibrate_baro(self: Type[T], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def factory_calibrate_temperature(self: Type[T], inplace: bool = False) -> T:
        pass

    @call_dataset()
    def cut_to_syncregion(self: Type[T], end: bool = False, warn_thres: Optional[int] = 30, inplace: bool = False) -> T:
        pass

    @call_dataset()
    def cut(self: Type[T], start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        pass

    @call_dataset()
    def cut_counter_val(self: Type[T], start: Optional[int] = None, stop: Optional[int] = None,
                        step: Optional[int] = None,
                        inplace: bool = False) -> T:
        pass

    @call_dataset()
    def downsample(self: Type[T], factor: int, inplace: bool = False) -> T:
        pass

    @call_dataset()
    def data_as_df(self, datastreams: Optional[Sequence[str]] = None, index: Optional[str] = None) -> pd.DataFrame:
        pass

    @call_dataset()
    def imu_data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        pass

    @call_dataset()
    def find_closest_calibration(self,
                                 folder: Optional[path_t] = None,
                                 recursive: bool = False,
                                 filter_cal_type: Optional[str] = None,
                                 before_after: Optional[str] = None):
        pass

    @call_dataset()
    def find_calibrations(self, folder: Optional[path_t] = None,
                          recursive: bool = True,
                          filter_cal_type: Optional[str] = None):
        pass
