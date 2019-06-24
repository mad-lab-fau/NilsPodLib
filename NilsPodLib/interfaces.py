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


class AnnotFieldMeta(type):
    def __new__(mcs, name, bases, attrs):  # noqa: N804
        cls = super().__new__(mcs, name, bases, attrs)
        if not bases:
            setattr(cls, '_base_keys', list(cls.__annotations__.keys()))
        elif getattr(cls, '_base_keys', None):
            keys = set(cls._base_keys) - set(attrs.keys())
            keys -= set(k for k in keys if k.startswith('_'))
            setattr(cls, '_base_keys', list(keys))
        return cls


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


class MultiDataset(metaclass=AnnotFieldMeta):
    path: path_t
    acc: Optional[Tuple['Datastream']] = None
    gyro: Optional[Tuple['Datastream']] = None
    mag: Optional[Tuple['Datastream']] = None
    baro: Optional[Tuple['Datastream']] = None
    analog: Optional[Tuple['Datastream']] = None
    ecg: Optional[Tuple['Datastream']] = None
    ppg: Optional[Tuple['Datastream']] = None
    temperature: Optional[Tuple['Datastream']] = None
    counter: Tuple[np.ndarray]

    size: Tuple[int]
    datastreams: Tuple[Iterable['Datastream']]

    ACTIVE_SENSORS: Tuple[Tuple[str]]

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

    def __getattribute__(self, name: str) -> Any:
        if name != '_base_keys' and name in self._base_keys:
            try:
                return self._cascading_dataset_attribute_access(name)
            except NotImplementedError:
                return super().__getattribute__(name)
        else:
            return super().__getattribute__(name)

    def _cascading_dataset_method_called(self, name: str, *args, **kwargs) -> Any:
        raise NotImplementedError('Implement either the method itself or _cascading_dataset_method_called to handle'
                                  'all method calls.')

    def _cascading_dataset_attribute_access(self, name: str) -> Any:
        raise NotImplementedError('Implement either the method itself to handle all attribute access.')
