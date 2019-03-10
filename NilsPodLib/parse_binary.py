#!/usr/bin/python3
# -*- coding: utf-8 -*-

import struct

import numpy as np

from NilsPodLib.header import Header


def int16_t(b, a):
    num = int(b) + (int(a) << 8)
    if num >= 0x8000:
        num -= 0xFFFF
    return num


def uint16_t(a, b):
    num = int(b) + (int(a) << 8)
    return num


def uint32_t(a, b, c, d):
    num = int(a) + (int(b) << 8) + (int(c) << 16) + (int(d) << 24)
    return num


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


def parse_binary(path):
    with open(path, 'rb') as f:
        data = f.read()

    HEADER_SIZE = data[0]
    print('Header Size = ' + str(HEADER_SIZE)) # TODO: Move to logging

    data = bytearray(data)
    header_bytes = np.asarray(struct.unpack(str(HEADER_SIZE) + 'b', data[0:HEADER_SIZE]), dtype=np.uint8)
    session_header = Header(header_bytes[1:HEADER_SIZE])

    PACKET_SIZE = session_header.packetSize

    data = read_binary_file_uint8(path, PACKET_SIZE, HEADER_SIZE)
    data = data.astype(np.uint32)

    idx = 0
    if session_header.gyroEnabled and session_header.accEnabled:
        gyr_data = np.zeros((len(data), 3))
        gyr_data[:, 0] = ((data[:, 0]) + (data[:, 1] << 8)).astype(np.int16)
        gyr_data[:, 1] = ((data[:, 2]) + (data[:, 3] << 8)).astype(np.int16)
        gyr_data[:, 2] = ((data[:, 4]) + (data[:, 5] << 8)).astype(np.int16)
        idx = idx + 6
        acc_data = np.zeros((len(data), 3))
        acc_data[:, 0] = ((data[:, 6]) + (data[:, 7] << 8)).astype(np.int16)
        acc_data[:, 1] = ((data[:, 8]) + (data[:, 9] << 8)).astype(np.int16)
        acc_data[:, 2] = ((data[:, 10]) + (data[:, 11] << 8)).astype(np.int16)
        idx = idx + 6
    elif session_header.accEnabled:
        acc_data = np.zeros((len(data), 3))
        acc_data[:, 0] = ((data[:, 0]) + (data[:, 1] << 8)).astype(np.int16)
        acc_data[:, 1] = ((data[:, 2]) + (data[:, 3] << 8)).astype(np.int16)
        acc_data[:, 2] = ((data[:, 4]) + (data[:, 5] << 8)).astype(np.int16)
        idx = idx + 6
        gyr_data = np.zeros(len(data))
    elif session_header.gyroEnabled:
        gyr_data = np.zeros((len(data), 3))
        gyr_data[:, 0] = ((data[:, 0]) + (data[:, 1] << 8)).astype(np.int16)
        gyr_data[:, 1] = ((data[:, 2]) + (data[:, 3] << 8)).astype(np.int16)
        gyr_data[:, 2] = ((data[:, 4]) + (data[:, 5] << 8)).astype(np.int16)
        idx = idx + 6
        acc_data = np.zeros(len(data))
    else:
        gyr_data = np.zeros(len(data))
        acc_data = np.zeros(len(data))

    if session_header.baroEnabled:
        baro = (data[:, idx] + (data[:, idx + 1] << 8)).astype(np.int16)
        baro = (baro + 101325) / 100.0
        idx = idx + 2
    else:
        baro = np.zeros(len(data))

    if session_header.pressureEnabled:
        pressure = data[:, idx:idx + 3].astype(np.uint8)
        idx = idx + 3
    else:
        pressure = np.zeros(len(data))

    if session_header.batteryEnabled:
        battery = (data[:, 17] * 2.0) / 100.0
        idx = idx + 1
    else:
        battery = np.zeros(len(acc_data))

    if (header_bytes[-3] == 1) and (header_bytes[-2] == 1):
        counter = data[:, -1] + (data[:, -2] << 8) + (data[:, -3] << 16) + (data[:, -4] << 24)
        sync = np.copy(counter)
        counter = np.bitwise_and(counter, 0x7FFFFFFF)

        sync = np.bitwise_and(sync, 0x80000000)
        sync = np.right_shift(sync, 31)

    else:
        counter = data[:, -1] + (data[:, -2] << 8) + (data[:, -3] << 16)
        sync = np.copy(counter)
        counter = np.bitwise_and(counter, 0x7FFFFFFF)

        sync = np.bitwise_and(sync, 0x80000000)
        sync = np.right_shift(sync, 23)

    if "V2.1" in session_header.versionFW:
        print("Firmware Version 2.1.x found") # TODO: Move to logging
        counter = data[:, -1] + (data[:, -2] << 8) + (data[:, -3] << 16) + (data[:, -4] << 24)
        sync = np.copy(counter)
        counter = np.bitwise_and(counter, 0x7FFFFFFF)

        sync = np.bitwise_and(sync, 0x80000000)
        sync = np.right_shift(sync, 31)

    return [acc_data, gyr_data, baro, pressure, battery, counter, sync, session_header]
