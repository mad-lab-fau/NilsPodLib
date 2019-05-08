"""Legacy support helper to convert older NilsPod files into new versions."""
from typing import NoReturn
import numpy as np

from NilsPodLib.utils import path_t


def convert_11_2(in_path: path_t, out_path: path_t) -> NoReturn:
    pass


def fix_little_endian_counter(data_bytes, packet_size):
    expected_samples = len(data_bytes) // packet_size
    data_bytes = data_bytes[:expected_samples * packet_size]
    data = np.reshape(data_bytes, (expected_samples, int(packet_size)))
    data[:, -4:] = np.flip(data[:, -4:], axis=-1)
    return data
