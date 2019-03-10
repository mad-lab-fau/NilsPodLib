#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import numpy as np


class Header:
    # default Session Header
    syncRole = 'disabled'
    samplingRate_Hz = 200
    accRange_G = 16
    gyroRange_dps = 2000
    syncDistance_ms = 0
    datetime_start = 0
    datetime_stop = 0
    unixTime_start = 0
    unixTime_stop = 0
    versionFW = 0
    sensorPosition = 'undefined'
    gyroEnabled = True
    accEnabled = True
    pressureEnabled = True
    baroEnabled = True
    batteryEnabled = True
    packetSize = 20
    lowVoltageTermination = False
    metaData = np.zeros(4)
    numSamples = 0

    SENSOR_FLAGS = {
        'gyroEnabled': 0x01,
        'pressureEnabled': 0x02,
        'baroEnabled': 0x04,
        'batteryEnabled': 0x08
    }

    SAMPLING_RATES = {
        10: 102.4,
        5: 204.8,
        4: 256.0,
        2: 512.0,
        1: 1024.0
    }

    SYNC_ROLE = {
        1: 'slave',
        2: 'master'
    }

    SENSOR_POS = {
        1: 'left foot',
        2: 'right foot',
        3: 'hip'
    }

    def __init__(self, header_packet=None):
        if header_packet:
            self.packetSize = header_packet[0]
            sensors = header_packet[1]

            for para, val in self.SENSOR_FLAGS.items():
                setattr(self, para, bool(sensors & val))

            self.samplingRate_Hz = self.SAMPLING_RATES[header_packet[2] & 0x0F]
            self.samplingTime_ms = (1.0 / self.samplingRate_Hz) * 1000.0

            self.lowVoltageTermination = bool(header_packet[2] & 0x80)

            self.syncRole = self.SYNC_ROLE.get(header_packet[3], self.syncRole)

            self.syncDistance_ms = header_packet[4] * 100.0

            self.accRange_G = header_packet[6]

            self.gyroRange_dps = header_packet[7] * 125

            self.sensorPosition = self.SENSOR_POS.get(header_packet[8], self.sensorPosition)

            self.metaData = header_packet[9:13]

            packed_date_time = int(header_packet[13]) | (int(header_packet[14]) << 8) | (
                        int(header_packet[15]) << 16) | (
                                       int(header_packet[16]) << 24)
            self.unixTime_start = packed_date_time
            self.datetime_start = datetime.datetime.fromtimestamp(self.unixTime_start)

            packed_date_time = int(header_packet[17]) | (int(header_packet[18]) << 8) | (
                        int(header_packet[19]) << 16) | (
                                       int(header_packet[20]) << 24)
            self.unixTime_stop = packed_date_time
            self.datetime_stop = datetime.datetime.fromtimestamp(self.unixTime_stop)

            self.numSamples = int(header_packet[21]) | (int(header_packet[22]) << 8) | (
                        int(header_packet[23]) << 16) | (
                                      int(header_packet[24]) << 24)

            self.versionFW = "V" + str(int(header_packet[-3])) + "." + str(int(header_packet[-2])) + "." + str(
                int(header_packet[-1]))
