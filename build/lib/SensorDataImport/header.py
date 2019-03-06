#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import datetime


class header:
    syncRole = None;
    samplingRate_Hz = None;
    accRange_G = None;
    gyroRange_dps = None;
    syncDistance_ms = None;
    timeStamp = None;
    versionFW = None;
    sensorPosition = None;
    imuEnabled = None;
    pressureEnabled = None;
    baroEnabled = None;
    batteryEnabled = None;
    lowVoltageTermination = None;        
    
    def __init__(self, headerPacket=None):
        if(headerPacket is None):
            #default session header
            self.syncRole = 'disabled';
            self.samplingRate_Hz = 200;
            self.accRange_G = 16;
            self.gyroRange_dps = 2000;
            self.syncDistance_ms = 0;
            self.timeStamp = 0;
            self.versionFW = 0;
            self.sensorPosition = 'undefined';
            self.imuEnabled = True;
            self.pressureEnabled = True;
            self.baroEnabled = True;
            self.batteryEnabled = True;
            self.lowVoltageTermination = False;
        else:
            
            sensors = headerPacket[1]
            if(sensors & 0x01):
                self.imuEnabled = True;
            else:
                self.imuEnabled = False;
            
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
    
            try:
                packedDateTime = int(headerPacket[13]) | (int(headerPacket[14]) << 8) | (int(headerPacket[15]) << 16) | (int(headerPacket[16]) << 24)
                S = packedDateTime & 0x3F;
                M = (packedDateTime >> 6) & 0x3F;
                H = (packedDateTime >> 12) & 0x1F;
                d = (packedDateTime >> 17) & 0x1F;
                m = (packedDateTime >> 22) & 0x0F;
                y = (packedDateTime >> 26) & 0x3F;
                self.timeStamp = datetime.datetime(hour=H, minute=M, second=S, day = d, month = m, year = 2000+y)
            except: # catch *all* exceptions
                print("Error while extracting timestamp! Using default timestamp: 00:00:00/01.01.2017")
                self.timeStamp = datetime.datetime(hour=0, minute=0, second=0, day = 1, month = 1, year = 2017)
    
            self.versionFW = "V" + str(int(headerPacket[17]) )+ "." + str(int(headerPacket[18])) + "." + str(int(headerPacket[19]))

        
        
        