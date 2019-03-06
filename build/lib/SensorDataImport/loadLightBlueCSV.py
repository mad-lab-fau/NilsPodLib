#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 10:56:42 2017

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
    return int(num)

def uint32_t(a,b,c,d):
    num = long(a) + (long(b) << 8) + (long(c) << 16) + (long(d) << 24)
    #num = int('{:08b}'.format(num)[::-1], 2)
    return num

import csv
import numpy as np
import pandas as pd

def loadLightBlueCSV(path):
    filereader = csv.reader(open(path))
    data = list();
    hexArray = list();
    row = iter(filereader)

    next(row) # skip header line
    next(row) # skip header line
 
    
    
    for row in filereader:
        tmpRow = row[3].split('x')[1] # extract the pure hex-string
        if(len(tmpRow) > 40):
            hexArray.append(tmpRow[0:40])
            hexArray.append(tmpRow[40:80])
            hexArray.append(tmpRow[80:120])
            hexArray.append(tmpRow[120:160])
            hexArray.append(tmpRow[160:200])
        elif (len(tmpRow) > 10):
            hexArray.append(tmpRow)
        
    # read csv-file and convert "hex strings" to uint_8 values
    for row in hexArray:
        tmpRow = row
       
        byteArray = list();
        for i in range(0,int(len(tmpRow)/2)):
            num = int(tmpRow[i*2]+tmpRow[i*2+1],16)
            byteArray.append(num)
        data.append(byteArray)
        
        
        
    accData = np.empty((len(data),3,))
    gyrData = np.empty((len(data),3,))
    force = np.empty((len(data),3,))
    
    counter = np.empty((len(data),1,))
    sync = np.empty((len(data),1,))
    battery = np.empty((len(data),1,))
    pressure = np.empty((len(data),1,))
    
    accX_offset = 0# 0.00020211724906677727
    accY_offset = 0#0.0005592478840726117
    accZ_offset = 0#1.060693733097422
    
    gyrX_offset = 0#0.2661586246240744
    gyrY_offset = 0#0.042934101291066964
    gyrZ_offset = 0#-1.3237904516868
    
    gyroScaleFactor = 1#16.4
    accScaleFactor =  1#2048.0
    
    k = 0;
    for row in data:
        
        gyrData[k][0] = (int16_t(row[0],row[1])/accScaleFactor)-accX_offset
        gyrData[k][1] = (int16_t(row[2],row[3])/accScaleFactor)- accY_offset
        gyrData[k][2] = (int16_t(row[4],row[5])/accScaleFactor)- accZ_offset
        
        accData[k][0] = (int16_t(row[6],row[7])/gyroScaleFactor)- gyrX_offset
        accData[k][1] = (int16_t(row[8],row[9])/gyroScaleFactor)- gyrY_offset
        accData[k][2] = (int16_t(row[10],row[11])/gyroScaleFactor)- gyrZ_offset
        
        
        pressure[k] = (int16_t(row[12],row[13]) + 101325)/100.0;
        
        force[k][0] = row[14];
        force[k][1] = row[15];
        force[k][2] = row[16];
        
        battery[k] = (row[17]*2.0)/100.0
         
        counter[k] = uint16_t(row[18],row[19]) & 0x7FFF;
        sync[k] = (uint16_t(row[18],row[19]) & 0x8000)>> 15
        #timestamp[k] = uint16_t(row[18],row[19]);#/4000000;
        
        #timestamp[k] = 10/1000
        
        k += 1
        
    return [accData, gyrData, pressure,force, battery, counter, sync]



def loadEgaitData(path):
    data = pd.read_csv(path);
    data = data.as_matrix();
    accData = -data[:,0:3]
    a1 = np.copy(accData[:, 1])
    a2 = np.copy(accData[:, 2])
    accData[:,1] = a2
    accData[:,2] = a1
    gyrData = -data[:,3:6]
    g1 = np.copy(gyrData[:, 1])
    g2 = np.copy(gyrData[:, 2])
    gyrData[:,1] = g2
    gyrData[:,2] = g1
    
    
    l = len(gyrData)
    timestamp = np.full((l, 1), 1/102.4)
    pressure = np.full((l, 1), 0)
    counter = np.full((l, 1), 0)
    
    return [accData, gyrData, pressure, timestamp, counter]