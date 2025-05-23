"""Set of helper functions used throughout the library."""

import copy
import datetime
import struct
import warnings
from pathlib import Path
from typing import Any, TypeVar

import numpy as np
import pytz
from packaging.version import Version

from nilspodlib.exceptions import CorruptedPackageWarning

path_t = TypeVar("path_t", str, Path)  # noqa: invalid-name
T = TypeVar("T")


def convert_little_endian(byte_list: np.ndarray, dtype: Any = int) -> np.ndarray:
    """Convert a little endian bytestring into a readable format.

    Parameters
    ----------
    byte_list :
        The array of bytes
    dtype :
        The datatype, the final value should be converted to.

    """
    byte_list = np.array(byte_list).astype(np.uint32)
    number = byte_list[0]
    for i, v in enumerate(byte_list[1:]):
        number |= v << int(8 * (i + 1))
    return number.astype(dtype)


def read_binary_uint8(data_bytes: np.ndarray, packet_size: int, expected_samples: int) -> np.ndarray:
    """Read a continuous stream of uint8 values into its separate datapoints.

    Parameters
    ----------
    data_bytes :
        The raw stream of data bytes
    packet_size :
        The size of each datapacket stored in the stream
    expected_samples :
        The expected number of samples in the data stream. This is only used to check the integrity
        of the datastream and raise a warning if, the number of samples in the dataset, does not match the expected
        number.

    Returns
    -------
    data
        Array with the shape (n_samples, packet_size)

    """
    packet_size = int(packet_size)
    expected_length = expected_samples * packet_size
    page_size = 2048
    if abs(len(data_bytes) - expected_length) > page_size // packet_size:
        warnings.warn(
            f"The provided binary file contains more or less than {page_size // packet_size} packages than "
            f"indicated by the header ({expected_samples} vs. {len(data_bytes) // packet_size}). This can be caused by "
            f"a bug affecting all synchronised sessions recorded with firmware versions before 0.14.0. \n"
            f"The full file will be read to avoid data loss, but this might add up to {page_size // packet_size} "
            f"corrupted packages at the end of the datastream.",
            CorruptedPackageWarning,
        )
        expected_length = (len(data_bytes) // packet_size) * packet_size

    elif expected_length > len(data_bytes):
        warnings.warn(
            "The provided binary file contains less samples than indicated by the header."
            " This might mean that the file was corrupted.",
            CorruptedPackageWarning,
        )
        expected_length = (len(data_bytes) // packet_size) * packet_size

    data_bytes = data_bytes[:expected_length]
    data = np.reshape(data_bytes, (expected_length // packet_size, packet_size))
    return data


def get_header_and_data_bytes(path: path_t) -> tuple[np.ndarray, np.ndarray]:
    """Separate a binary file into its header and data part."""
    with path.open(mode="rb") as f:
        header = f.read(1)
        header_size = header[0]
        header += f.read(header_size - 1)
        data_bytes = np.fromfile(f, dtype=np.dtype("B"))

    header = bytearray(header)
    header_bytes = np.asarray(struct.unpack(str(header_size) + "B", header[0:header_size]), dtype=np.uint8)

    return header_bytes, data_bytes


def get_sample_size_from_header_bytes(header_bytes: np.ndarray) -> int:
    """Get the size of an individual data sample (in bytes) from the header info."""
    return int(header_bytes[1])


def get_strict_version_from_header_bytes(header_bytes: np.ndarray) -> Version:
    """Extract the version number from a byte header."""
    return Version("{}.{}.{}".format(*(int(x) for x in header_bytes[-3:])))


def inplace_or_copy(obj: T, inplace: bool) -> T:
    """Either create a deepcopy of the object or return it based on the value of `inplace`."""
    if inplace is True:
        return obj
    return copy.deepcopy(obj)


def validate_existing_overlap(start_vals: np.ndarray, end_vals: np.ndarray) -> bool:
    """Check that multiple intervals indicated by their start and stop values do all have an overlapping region.

    Raises
    ------
    ValueError
        If any of the intervals are invalid, because their end values is before their start value.

    """
    if not all(i < j for i, j in zip(start_vals, end_vals, strict=False)):
        raise ValueError("The start values need to be smaller then their respective end values!")
    return np.max(start_vals) < np.min(end_vals)


def remove_docstring_indent(doc_str: str) -> str:
    """Remove the additional indent of a multiline docstring.

    This can be helpful, if docstrings are combined programmatically.
    """
    lines = doc_str.split("\n")
    if len(lines) <= 1:
        return doc_str
    first_non_summary_line = next(line for line in lines[1:] if line)
    indent = len(first_non_summary_line) - len(first_non_summary_line.lstrip())
    cut_lines = [lines[0]]
    for line in lines:
        cut_lines.append(line[indent:])
    return "\n".join(cut_lines)


def raise_timezone_error(timezone):
    """Raise a ValueError, if timezone is None."""
    if not timezone:
        raise ValueError(
            "Local datetime information is only available, if a timezone is specified for the recording. "
            "This can be done via the `tz` parameter during initialization."
        )


def convert_to_local_time(utc_datetime: datetime.datetime, timezone: str | None) -> datetime.datetime:
    """Convert a utc datetime object to a different timezone."""
    raise_timezone_error(timezone)
    return utc_datetime.astimezone(pytz.timezone(timezone))
