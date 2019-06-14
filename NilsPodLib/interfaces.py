from typing import Optional, Iterable, Tuple, Union, Any, TypeVar, TYPE_CHECKING, Type, Sequence, List

import numpy as np
import pandas as pd
from NilsPodLib.header import Header

from NilsPodLib.utils import path_t

T = TypeVar('T')

if TYPE_CHECKING:
    from NilsPodLib.datastream import Datastream  # noqa: F401
    from imucal import CalibrationInfo  # noqa: F401


class AnnotFieldMeta(type):
    def __new__(mcs, name, bases, attrs):  # noqa: N804
        cls = super().__new__(mcs, name, bases, attrs)
        if not bases:
            setattr(cls, '_' + name + '_fields', list(cls.__annotations__.keys()))
        return cls


class CascadingDatasetInterface(metaclass=AnnotFieldMeta):
    path: path_t
    acc: Optional['Datastream'] = None
    gyro: Optional['Datastream'] = None
    mag: Optional['Datastream'] = None
    baro: Optional['Datastream'] = None
    analog: Optional['Datastream'] = None
    ecg: Optional['Datastream'] = None
    ppg: Optional['Datastream'] = None
    temperature: Optional['Datastream'] = None
    counter: np.ndarray
    info: Header

    size: int
    datastreams: Iterable['Datastream']

    ACTIVE_SENSORS: Tuple[str]

    def calibrate_imu(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_imu', calibration, inplace)

    def calibrate_acc(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_imu', calibration, inplace)

    def calibrate_gyro(self: Type[T], calibration: Union['CalibrationInfo', path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_gyro', calibration, inplace)

    def factory_calibrate_imu(self: Type[T], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_imu', inplace)

    def factory_calibrate_gyro(self: Type[T], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_gyro', inplace)

    def factory_calibrate_baro(self: Type[T], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_baro', inplace)

    def factory_calibrate_temperature(self: Type[T], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_temperature', inplace)

    def cut_to_syncregion(self: Type[T], end: bool = False, warn_thres: Optional[int] = 30, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('cut_to_syncregion', end, warn_thres, inplace)

    def cut(self: Type[T], start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('cut', start, stop, step, inplace)

    def cut_counter_val(self: Type[T], start: Optional[int] = None, stop: Optional[int] = None,
                        step: Optional[int] = None,
                        inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('cut_counter_val', start, stop, step, inplace)

    def downsample(self: Type[T], factor: int, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('downsample', factor, inplace)

    def data_as_df(self, datastreams: Optional[Sequence[str]] = None, index: Optional[str] = None) -> pd.DataFrame:
        return self._cascading_dataset_method_called('data_as_df', datastreams, index)

    def imu_data_as_df(self, index: Optional[str] = None) -> pd.DataFrame:
        return self._cascading_dataset_method_called('imu_data_as_df', index)

    def find_closest_calibration(self,
                                 folder: Optional[path_t] = None,
                                 recursive: bool = False,
                                 filter_cal_type: Optional[str] = None,
                                 before_after: Optional[str] = None):
        return self._cascading_dataset_method_called('find_closest_calibration', folder, recursive, filter_cal_type,
                                                     before_after)

    def find_calibrations(self, folder: Optional[path_t] = None,
                          recursive: bool = True,
                          filter_cal_type: Optional[str] = None):
        return self._cascading_dataset_method_called('find_calibrations', folder, recursive, filter_cal_type)

    def __getattribute__(self, name: str) -> Any:
        if name != '_CascadingDatasetInterface_fields' \
                and name in self._CascadingDatasetInterface_fields:
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
