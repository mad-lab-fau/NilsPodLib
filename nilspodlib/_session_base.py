"""Internal bases for sessions to make it easier to call dataset methods on the session object.

@author: Arne Küderle
"""

from functools import wraps
from typing import Optional, Iterable, Tuple, TypeVar, TYPE_CHECKING, Type, Sequence

import numpy as np

from nilspodlib import Dataset
from nilspodlib.utils import path_t, inplace_or_copy, remove_docstring_indent

T = TypeVar("T")

if TYPE_CHECKING:
    from nilspodlib.datastream import Datastream  # noqa: F401
    import pandas as pd  # noqa: F401


class CascadingDatasetField:
    """A simple descriptor object to forward attribute access to all datasets of a session."""

    def __set_name__(self, owner, name):
        self.name = name
        self.__doc__ = getattr(Dataset, self.name, None).__doc__

    def __get__(self, instance, owner):
        return tuple(getattr(d, self.name) for d in instance.datasets)


def call_dataset(autogen_doc=True):  # noqa: D202
    """Forward all method calls to all datasets of a session.

    This function respects the inplace feature and will create a copy of the session object if required.

    Args:
        autogen_doc: If True, the docstring of the respective dataset method is copied to the method with short pretext.
            If a docstring already exists, the dataset docstring will be appended WITHOUT pretext.

    """

    def wrapped(method):
        @wraps(method)
        def cascading_access(*args, **kwargs):
            session = args[0]
            return_vals = tuple(getattr(d, method.__name__)(*args[1:], **kwargs) for d in session.datasets)

            if all(isinstance(d, Dataset) for d in return_vals):
                inplace = kwargs.get("inplace", False)
                s = inplace_or_copy(session, inplace)
                s.datasets = return_vals
                return s
            return return_vals

        if autogen_doc:
            if cascading_access.__doc__:
                cascading_access.__doc__ += "\n\n"
            else:
                cascading_access.__doc__ = (
                    "Apply `Dataset.{0}` to all datasets of the session.\n\n"
                    "See :py:meth:`nilspodlib.dataset.Dataset.{0}` for more details. "
                    "The docstring of this method is included below:\n\n".format(method.__name__)
                )
            cascading_access.__doc__ += remove_docstring_indent(getattr(Dataset, method.__name__).__doc__)
            cascading_access.__doc__ += "\nSee Also:\n" "   :py:meth:`nilspodlib.dataset.Dataset.{0}`".format(
                method.__name__
            )
        return cascading_access

    return wrapped


class _MultiDataset:
    """Wrapper that holds all attributes and methods that can be simply called on multiple datasets.

    Note:
        This class should not be used as public interface and is only relevant as base for the session class

    This class uses a decorator for methods and a descriptor for attributes to automatically forward all calls to
    multiple datasets.
    See the implementation of `CascadingDatasetField` and `call_dataset` for details.

    """

    path: path_t = CascadingDatasetField()
    acc: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    gyro: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    mag: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    baro: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    analog: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    ecg: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    ppg: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    temperature: Tuple[Optional["Datastream"]] = CascadingDatasetField()
    counter: Tuple[np.ndarray] = CascadingDatasetField()

    size: Tuple[int] = CascadingDatasetField()
    datastreams: Tuple[Iterable["Datastream"]] = CascadingDatasetField()

    ACTIVE_SENSORS: Tuple[Tuple[str]] = CascadingDatasetField()

    # This needs to be implemented by the session
    datasets: Tuple[Dataset]

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
    def cut(
        self: Type[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        inplace: bool = False,
    ) -> T:
        pass

    @call_dataset()
    def cut_counter_val(
        self: Type[T],
        start: Optional[int] = None,
        stop: Optional[int] = None,
        step: Optional[int] = None,
        inplace: bool = False,
    ) -> T:
        pass

    @call_dataset()
    def downsample(self: Type[T], factor: int, inplace: bool = False) -> T:
        pass

    @call_dataset()
    def data_as_df(
        self,
        datastreams: Optional[Sequence[str]] = None,
        index: Optional[str] = None,
        include_units: Optional[bool] = True,
    ) -> Tuple["pd.DataFrame"]:
        pass

    @call_dataset()
    def imu_data_as_df(self, index: Optional[str] = None) -> Tuple["pd.DataFrame"]:
        pass

    @call_dataset()
    def find_closest_calibration(
        self,
        folder: Optional[path_t] = None,
        recursive: bool = False,
        filter_cal_type: Optional[str] = None,
        before_after: Optional[str] = None,
        ignore_file_not_found: Optional[bool] = False,
    ):
        pass

    @call_dataset()
    def find_calibrations(
        self,
        folder: Optional[path_t] = None,
        recursive: bool = True,
        filter_cal_type: Optional[str] = None,
        ignore_file_not_found: Optional[bool] = False,
    ):
        pass
