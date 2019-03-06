# -*- coding: utf-8 -*-

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 19 15:45:53 2017

@author: nils
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 16:21:39 2017

@author: nils
"""

def int16_t(b,a):
    num = long(b) + (long(a) << 8)
    if num >= 0x8000:
        num -= 0xFFFF
    return num

def uint16_t(a,b):
    num = long(b) + (long(a) << 8)
    #num = int('{:08b}'.format(num)[::-1], 2)
    return num

def uint32_t(a,b,c,d):
    num = long(a) + (long(b) << 8) + (long(c) << 16) + (long(d) << 24)
    #num = int('{:08b}'.format(num)[::-1], 2)
    return num


import numpy as np
import matplotlib.pyplot as plt
from SensorDataImport.header import header




def readBinaryFile_uint8(path, packet_size, skipHeader):
    f = open(path, 'rb')
    if skipHeader:
        f.seek(20) # skip header bytes
    data = np.fromfile(f, dtype=np.dtype('B'))            
    #data = np.reshape(data,(len(data)/(packet_size/2),(packet_size/2)))
    data = data[0:(int(len(data)/packet_size)*packet_size)]
    data = np.reshape(data,(int(len(data)/(packet_size)),int(packet_size)))
    return data;

def readBinaryFile_int16(path,packet_size, skipHeader):
    f = open(path, 'rb')
    if skipHeader:
        f.seek(20) # skip header bytes
    data = np.fromfile(f, dtype=np.dtype('i2'))            
    #data = np.reshape(data,(len(data)/(packet_size/2),(packet_size/2)))
    data = data[0:(int(len(data)/int(packet_size/2))*int(packet_size/2))]
    data = np.reshape(data,(int(len(data)/(packet_size/2)),int(packet_size/2)))
    return data;

def loadBinary(path, headerPresent=1):
    if headerPresent:
        with open(path, 'rb') as f:
            PACKET_SIZE = f.read()[0]
    else:
         PACKET_SIZE = 20

    data_uint8 = readBinaryFile_uint8(path,PACKET_SIZE, 1);
    data_int16 = readBinaryFile_int16(path,PACKET_SIZE, 1);            
    
    #extract header information from first packet
    if headerPresent:
        headerPacket = readBinaryFile_uint8(path,20, 0);
        sessionHeader = header(headerPacket[0])
    else:
        sessionHeader = header(None);
          
    #cut away potential header packet
    data_uint8 = data_uint8[0:len(data_uint8)]
    data_int16 = data_int16[0:len(data_int16)]            

    idx = 0;
    if(sessionHeader.imuEnabled):
        gyrData = data_int16[:,0:3]
        accData = data_int16[:,3:6]
        idx = idx + 12;
    
    if(sessionHeader.baroEnabled):
        baro = (data_int16[:,6] + 101325)/100.0;
        idx = idx + 2;
    else:
        baro = np.zeros(len(accData));
    
    if(sessionHeader.pressureEnabled):
        pressure = data_uint8[:,14:17]
        idx = idx + 3;
    else:
        pressure = np.zeros(len(accData));
    
    if(sessionHeader.batteryEnabled):
        battery = (data_uint8[:,17]*2.0)/100.0
        idx = idx + 1;
    else:
        battery = np.zeros(len(accData));
    
    counter = data_int16[:,int(idx/2)]
    counter = counter.astype(dtype=np.dtype('u2'))
    counter = counter.byteswap();
    sync = np.copy(counter)
    counter = np.bitwise_and(counter,0x7FFF)
       
    sync = np.bitwise_and(sync, 0x8000);
    sync = np.right_shift(sync,15);
   
    
    if headerPresent:
        return [accData, gyrData, baro, pressure, battery, counter, sync, sessionHeader]
    else:
        return [accData, gyrData, baro, pressure, battery, counter, sync]
     