#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import numpy as np


class Header:
    syncRole = None
    samplingRate_Hz = None
    accRange_G = None
    gyroRange_dps = None
    syncDistance_ms = None
    datetime_stop = None
    datetime_start = None
    versionFW = None
    sensorPosition = None
    gyroEnabled = None
    accEnabled = True
    pressureEnabled = None
    baroEnabled = None
    batteryEnabled = None
    lowVoltageTermination = None
    packetSize = None
    metaData = None
    numSamples = None

    def __init__(self, header_packet=None):
        if header_packet is None:
            # default Session Header
            self.syncRole = 'disabled'
            self.samplingRate_Hz = 200
            self.accRange_G = 16
            self.gyroRange_dps = 2000
            self.syncDistance_ms = 0
            self.datetime_start = 0
            self.datetime_stop = 0
            self.unixTime_start = 0
            self.unixTime_stop = 0
            self.versionFW = 0
            self.sensorPosition = 'undefined'
            self.gyroEnabled = True
            self.accEnabled = True
            self.pressureEnabled = True
            self.baroEnabled = True
            self.batteryEnabled = True
            self.packetSize = 20
            self.lowVoltageTermination = False
            self.metaData = np.zeros(4)
            self.numSamples = 0
        else:

            self.packetSize = header_packet[0]
            sensors = header_packet[1]
            if sensors & 0x01:
                self.gyroEnabled = True
            else:
                self.gyroEnabled = False

            if sensors & 0x02:
                self.pressureEnabled = True
            else:
                self.pressureEnabled = False

            if sensors & 0x04:
                self.baroEnabled = True
            else:
                self.baroEnabled = False

            if sensors & 0x08:
                self.batteryEnabled = True
            else:
                self.batteryEnabled = False

            sr = header_packet[2] & 0x0F

            if sr == 10:
                self.samplingRate_Hz = 102.4
            elif sr == 5:
                self.samplingRate_Hz = 204.8
            elif sr == 4:
                self.samplingRate_Hz = 256.0
            elif sr == 2:
                self.samplingRate_Hz = 512.0
            elif sr == 1:
                self.samplingRate_Hz = 1024.0

            if header_packet[2] & 0x80:
                self.lowVoltageTermination = True
            else:
                self.lowVoltageTermination = False

            self.samplingTime_ms = (1.0 / self.samplingRate_Hz) * 1000.0

            if header_packet[3] == 2:
                self.syncRole = 'master'
            elif header_packet[3] == 1:
                self.syncRole = 'slave'
            else:
                self.syncRole = 'disabled'

            self.syncDistance_ms = header_packet[4] * 100.0

            self.accRange_G = header_packet[6]

            self.gyroRange_dps = header_packet[7] * 125

            if header_packet[8] == 1:
                self.sensorPosition = 'left foot'
            elif header_packet[8] == 2:
                self.sensorPosition = 'right foot'
            elif header_packet[8] == 3:
                self.sensorPosition = 'hip'
            else:
                self.sensorPosition = 'not defined'

            self.metaData = header_packet[9:13]

            packedDateTime = int(header_packet[13]) | (int(header_packet[14]) << 8) | (int(header_packet[15]) << 16) | (
                    int(header_packet[16]) << 24)
            self.unixTime_start = packedDateTime
            self.datetime_start = datetime.datetime.fromtimestamp(self.unixTime_start)

            packedDateTime = int(header_packet[17]) | (int(header_packet[18]) << 8) | (int(header_packet[19]) << 16) | (
                    int(header_packet[20]) << 24)
            self.unixTime_stop = packedDateTime
            self.datetime_stop = datetime.datetime.fromtimestamp(self.unixTime_stop)

            self.numSamples = int(header_packet[21]) | (int(header_packet[22]) << 8) | (int(header_packet[23]) << 16) | (
                    int(header_packet[24]) << 24)

            self.versionFW = "V" + str(int(header_packet[-3])) + "." + str(int(header_packet[-2])) + "." + str(
                int(header_packet[-1]))

    def convertUnixTimeToDateTime(self, unixTimeStamp):
        return datetime.datetime.fromtimestamp(unixTimeStamp)
