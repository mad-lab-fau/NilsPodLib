# -*- coding: utf-8 -*-

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime
import numpy as np

class headerFreeRTOS:
    syncRole = None;
    samplingRate_Hz = None;
    accRange_G = None;
    gyroRange_dps = None;
    syncDistance_ms = None;
    datetime_stop = None;
    datetime_start = None;
    versionFW = None;
    sensorPosition = None;
    gyroEnabled = None;
    accEnabled = True;
    pressureEnabled = None;
    baroEnabled = None;
    batteryEnabled = None;
    lowVoltageTermination = None;     
    packetSize = None;
    metaData = None;
    numSamples = None;
    
    def __init__(self, headerPacket=None):
        if(headerPacket is None):
            #default session header
            self.syncRole = 'disabled';
            self.samplingRate_Hz = 200;
            self.accRange_G = 16;
            self.gyroRange_dps = 2000;
            self.syncDistance_ms = 0;
            self.datetime_start = 0;
            self.datetime_stop = 0;
            self.unixTime_start = 0;
            self.unixTime_stop = 0;
            self.versionFW = 0;
            self.sensorPosition = 'undefined';
            self.gyroEnabled = True;
            self.accEnabled = True;
            self.pressureEnabled = True;
            self.baroEnabled = True;
            self.batteryEnabled = True;
            self.packetSize = 20;
            self.lowVoltageTermination = False;
            self.metaData = np.zeros(4)
            self.numSamples = 0
        else:
            
            self.packetSize = headerPacket[0]
            sensors = headerPacket[1]
            if(sensors & 0x01):
                self.gyroEnabled = True;
            else:
                self.gyroEnabled = False;
            
            if(sensors & 0x02):
                self.pressureEnabled = True;
            else:
                self.pressureEnabled = False;
                
            if(sensors & 0x04):
                self.baroEnabled = True;
            else:
                self.baroEnabled = False;
                
            if(sensors & 0x08):
                self.batteryEnabled = True;
            else:
                self.batteryEnabled = False;
            
            self.samplingTime_ms = headerPacket[2] & 0x7F;
            if(headerPacket[2] & 0x80):
                self.lowVoltageTermination  = True;
            else:
                self.lowVoltageTermination = False;
            
            self.samplingRate_Hz = 1.0/ (self.samplingTime_ms*1e-3)
        
            if headerPacket[3] == 2:
                self.syncRole = 'master'
            elif headerPacket[3] == 1:
                self.syncRole = 'slave'
            else:
                self.syncRole = 'disabled'
            
        
            self.syncDistance_ms = headerPacket[4]*100.0
        
            self.accRange_G = headerPacket[6]
        
            self.gyroRange_dps = headerPacket[7]*125;
        
            if headerPacket[8] == 1:
                self.sensorPosition = 'left foot'
            elif headerPacket[8] == 2:
                self.sensorPosition = 'right foot'
            elif headerPacket[8] == 3:
                self.sensorPosition = 'hip'
            else:
                self.sensorPosition = 'not defined'
                
            self.metaData = headerPacket[9:13];
    
            packedDateTime = int(headerPacket[13]) | (int(headerPacket[14]) << 8) | (int(headerPacket[15]) << 16) | (int(headerPacket[16]) << 24)
            self.unixTime_start = packedDateTime;
            self.datetime_start = datetime.datetime.fromtimestamp(self.unixTime_start)
                
            packedDateTime = int(headerPacket[17]) | (int(headerPacket[18]) << 8) | (int(headerPacket[19]) << 16) | (int(headerPacket[20]) << 24)
            self.unixTime_stop  = packedDateTime;
            self.datetime_stop = datetime.datetime.fromtimestamp(self.unixTime_stop)
            
            self.numSamples = int(headerPacket[21]) | (int(headerPacket[22]) << 8) | (int(headerPacket[23]) << 16) | (int(headerPacket[24]) << 24)
    
            self.versionFW = "V" + str(int(headerPacket[-3]) )+ "." + str(int(headerPacket[-2])) + "." + str(int(headerPacket[-1]))
            
    def convertUnixTimeToDateTime(self,unixTimeStamp):
        return datetime.datetime.fromtimestamp(unixTimeStamp);

        
        
        