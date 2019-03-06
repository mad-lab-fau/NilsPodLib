
import numpy as np
import matplotlib.pyplot as plt
from NilsPodLib.header import header
import struct

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

def readBinaryFile_uint8(path, packet_size, skipHeaderBytes):
    f = open(path, 'rb')
    f.seek(skipHeaderBytes) # skip header bytes
    data = np.fromfile(f, dtype=np.dtype('B'))            
    #data = np.reshape(data,(len(data)/(packet_size/2),(packet_size/2)))
    data = data[0:(int(len(data)/packet_size)*packet_size)]
    data = np.reshape(data,(int(len(data)/(packet_size)),int(packet_size)))
    return data;

def readBinaryFile_int16(path,packet_size, skipHeaderBytes):
    f = open(path, 'rb')
    f.seek(skipHeaderBytes) # skip header bytes
    data = np.fromfile(f, dtype=np.dtype('i2'))            
    #data = np.reshape(data,(len(data)/(packet_size/2),(packet_size/2)))
    data = data[0:(int(len(data)/int(packet_size/2))*int(packet_size/2))]
    data = np.reshape(data,(int(len(data)/(packet_size/2)),int(packet_size/2)))
    return data;    

def parseBinary(path):
    f = open(path, 'rb')
    f = f.read();
    HEADER_SIZE = f[0];
    print('Header Size = ' +str(HEADER_SIZE))
    f = bytearray(f)
    headerBytes = np.asarray(struct.unpack(str(HEADER_SIZE)+'b',f[0:HEADER_SIZE]),dtype=np.uint8)
    sessionHeader = header(headerBytes[1:HEADER_SIZE])
  
    
    PACKET_SIZE = sessionHeader.packetSize;
    
    data = readBinaryFile_uint8(path,PACKET_SIZE, HEADER_SIZE);
    data = data.astype(np.uint32)
   
    idx = 0;
    if(sessionHeader.gyroEnabled and sessionHeader.accEnabled):
        gyrData = np.zeros((len(data),3))
        gyrData[:,0] = ((data[:,0]) + (data[:,1] << 8)).astype(np.int16)
        gyrData[:,1] = ((data[:,2]) + (data[:,3] << 8)).astype(np.int16)
        gyrData[:,2] = ((data[:,4]) + (data[:,5] << 8)).astype(np.int16)
        idx = idx + 6;
        accData = np.zeros((len(data),3))
        accData[:,0] = ((data[:,6]) + (data[:,7] << 8)).astype(np.int16)
        accData[:,1] = ((data[:,8]) + (data[:,9] << 8)).astype(np.int16)
        accData[:,2] = ((data[:,10]) + (data[:,11] << 8)).astype(np.int16)
        idx = idx + 6;
    elif(sessionHeader.accEnabled):
        accData = np.zeros((len(data),3))
        accData[:,0] = ((data[:,0]) + (data[:,1] << 8)).astype(np.int16)
        accData[:,1] = ((data[:,2]) + (data[:,3] << 8)).astype(np.int16)
        accData[:,2] = ((data[:,4]) + (data[:,5] << 8)).astype(np.int16)
        idx = idx + 6;
        gyrData = np.zeros(len(data));
    elif(sessionHeader.gyroEnabled):
        gyrData = np.zeros((len(data),3))
        gyrData[:,0] = ((data[:,0]) + (data[:,1] << 8)).astype(np.int16)
        gyrData[:,1] = ((data[:,2]) + (data[:,3] << 8)).astype(np.int16)
        gyrData[:,2] = ((data[:,4]) + (data[:,5] << 8)).astype(np.int16)
        idx = idx + 6;
        accData = np.zeros(len(data));
    else:
        gyrData = np.zeros(len(data));
        accData = np.zeros(len(data))
        

        
    if(sessionHeader.baroEnabled):
        baro = np.zeros(len(data));
        baro = (data[:,idx] + (data[:,idx+1] << 8)).astype(np.int16)
        baro = (baro + 101325)/100.0;
        idx = idx + 2;
    else:
        baro = np.zeros(len(data));
    
    if(sessionHeader.pressureEnabled):
        pressure = data[:,idx:idx+3].astype(np.uint8)
        idx = idx + 3;
    else:
        pressure = np.zeros(len(data));
    
    if(sessionHeader.batteryEnabled):
        battery = (data[:,17]*2.0)/100.0
        idx = idx + 1;
    else:
        battery = np.zeros(len(accData));
    
    if((headerBytes[-3] == 1) and (headerBytes[-2] == 1)):
        counter =  data[:,-1] + (data[:,-2] << 8) + (data[:,-3] << 16) + (data[:,-4] << 24);   
        sync = np.copy(counter)
        counter = np.bitwise_and(counter,0x7FFFFFFF)
       
        sync = np.bitwise_and(sync, 0x80000000);
        sync = np.right_shift(sync,31);
    
    else:
        counter =  data[:,-1] + (data[:,-2] << 8) + (data[:,-3] << 16);   
        sync = np.copy(counter)
        counter = np.bitwise_and(counter,0x7FFFFFFF)
       
        sync = np.bitwise_and(sync, 0x80000000);
        sync = np.right_shift(sync,23);
        
    if("V2.1" in sessionHeader.versionFW):
        print("Firmware Version 2.1.x found")
        counter =  data[:,-1] + (data[:,-2] << 8) + (data[:,-3] << 16) + (data[:,-4] << 24);   
        sync = np.copy(counter)
        counter = np.bitwise_and(counter,0x7FFFFFFF)
       
        sync = np.bitwise_and(sync, 0x80000000);
        sync = np.right_shift(sync,31);
   
    return [accData, gyrData, baro, pressure, battery, counter, sync, sessionHeader]
        
    
         