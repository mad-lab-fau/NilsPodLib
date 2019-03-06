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
import os
import matplotlib.pyplot as plt
from DataHandling.InSoleHeader import InSoleHeader

def readBinaryFile(path, packet_size):
    #### converting binary sensor data file to 
    file_size = os.path.getsize(path)
    
    dataBin = np.zeros(shape=((file_size/packet_size)+1,packet_size))
    
    i = 0
    j = 0
    k = 0
    
    with open(path) as f:
        s = f.read()
        for c in s:
            dataBin[j][i] = ord(c)
            i = i+1
            k = k + 1
            if i == packet_size:
                i = 0
                j = j+1
    
    return dataBin;

def readBinaryFile_uint8(path, packet_size):
    dataBin = np.fromfile(path, dtype=np.dtype('B'))            
    dataBin = np.reshape(dataBin,(len(dataBin)/packet_size,packet_size))
    return dataBin;


def loadAndroidBin(path, header=1):

   
    if header:
        with open(path, 'rb') as f:
            PACKET_SIZE = ord(f.read()[0])
    else:
         PACKET_SIZE = 20

    dataBin = readBinaryFile_uint8(path,PACKET_SIZE);            
    
    #extract header information from first packet
    if header:
        sessionHeader = InSoleHeader(dataBin[-1,:])
        
       
    #cut away potential header packet
    dataBin = dataBin[1:len(dataBin)]        

    accData = np.empty((len(dataBin),3,))
    gyrData = np.empty((len(dataBin),3,))
    force = np.empty((len(dataBin),3,))
    
    counter = np.empty((len(dataBin),1,))
    sync = np.empty((len(dataBin),1,))
    battery = np.empty((len(dataBin),1,))
    pressure = np.empty((len(dataBin),1,))
    tmp = 0
    
    for i in range(0,len(dataBin)):
        gyrData[i][0] = (int16_t(dataBin[i][0],dataBin[i][1]))
        gyrData[i][1] = (int16_t(dataBin[i][2],dataBin[i][3]))
        gyrData[i][2] = (int16_t(dataBin[i][4],dataBin[i][5]))
        
        accData[i][0] = (int16_t(dataBin[i][6],dataBin[i][7]))
        accData[i][1] = (int16_t(dataBin[i][8],dataBin[i][9]))
        accData[i][2] = (int16_t(dataBin[i][10],dataBin[i][11]))
        
        pressure[i] = (int16_t(dataBin[i][12],dataBin[i][13]) + 101325)/100.0;
         
        force[i][0]  = dataBin[i][14];
        force[i][1]  = dataBin[i][15];
        force[i][2]  = dataBin[i][16];
            
        battery[i] = (dataBin[i][17]*2.0)/100.0
        
        tmp = uint16_t(dataBin[i][18],dataBin[i][19])
        
        counter[i] = long(tmp & 0x7FFF)
        sync[i] = (tmp & 0x8000) >> 15;
       
        
    badStart = 0;
    gyrData = gyrData[badStart:len(gyrData)-1,:]
    accData = accData[badStart:len(accData)-1,:]
    pressure = pressure[badStart:len(pressure)-1,:]
    force = force[badStart:len(force)-1,:]
    battery = battery[badStart:len(battery)-1,:]
    counter = counter[badStart:len(counter)-1,:]
    sync = sync[badStart:len(sync)-1,:]
    
    
    if header:
        return [accData, gyrData, pressure,force, battery, counter, sync, sessionHeader]
    else:
        return [accData, gyrData, pressure,force, battery, counter, sync]
     