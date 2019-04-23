#!/usr/bin/python3
# -*- coding: utf-8 -*-

from typing import TypeVar

import numpy as np
from pathlib import Path

path_t = TypeVar('path_t', str, Path)


def convert_little_endian(byte_list, dtype=int):
    byte_list = np.array(byte_list).astype(dtype)
    number = byte_list[0]
    for i, v in enumerate(byte_list[1:]):
        number |= v << int(8*(i+1))
    return number


def read_binary_file_uint8(path, packet_size, skipHeaderBytes):
    with open(path, 'rb') as f:
        f.seek(skipHeaderBytes)  # skip Header bytes
        data = np.fromfile(f, dtype=np.dtype('B'))
    data = data[0:(int(len(data) / packet_size) * packet_size)]
    data = np.reshape(data, (int(len(data) / packet_size), int(packet_size)))
    return data


def read_binary_file_int16(path, packet_size, skipHeaderBytes):
    with open(path, 'rb') as f:
        f.seek(skipHeaderBytes)  # skip Header bytes
        data = np.fromfile(f, dtype=np.dtype('i2'))
    data = data[0:(int(len(data) / int(packet_size / 2)) * int(packet_size / 2))]
    data = np.reshape(data, (int(len(data) / (packet_size / 2)), int(packet_size / 2)))
    return data
