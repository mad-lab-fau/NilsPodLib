from typing import Optional, Iterable, Tuple, Union, Any, TypeVar, TYPE_CHECKING

import numpy as np
import pandas as pd
from imucal import CalibrationInfo

from NilsPodLib.header import Header
from NilsPodLib.utils import path_t

T = TypeVar('T')

if TYPE_CHECKING:
    from NilsPodLib.datastream import Datastream


class AnnotFieldMeta(type):
    def __new__(mcs, name, bases, attrs):
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
    battery: Optional['Datastream'] = None
    counter: np.ndarray
    info: Header

    size: int
    datastreams: Iterable['Datastream']

    ACTIVE_SENSORS: Tuple[str]

    def calibrate_imu(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_imu', calibration, inplace)

    def calibrate_acc(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_imu', calibration, inplace)

    def calibrate_gyro(self: T, calibration: Union[CalibrationInfo, path_t], inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('calibrate_gyro', calibration, inplace)

    def factory_calibrate_imu(self: T, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_imu', inplace)

    def factory_calibrate_gyro(self: T, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_gyro', inplace)

    def factory_calibrate_baro(self: T, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_baro', inplace)

    def factory_calibrate_battery(self: T, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('factory_calibrate_battery', inplace)

    def cut_to_syncregion(self: T, inplace=False) -> T:
        return self._cascading_dataset_method_called('cut_to_syncregion', inplace)

    def cut(self: T, start: Optional[int] = None, stop: Optional[int] = None, step: Optional[int] = None,
            inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('cut', start, stop, step)

    def downsample(self: T, factor: int, inplace: bool = False) -> T:
        return self._cascading_dataset_method_called('downsample', factor, inplace)

    def data_as_df(self) -> pd.DataFrame:
        return self._cascading_dataset_method_called('data_as_df')

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


