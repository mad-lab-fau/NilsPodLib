#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 28 11:32:22 2017

@author: nils
"""

import numpy as np
from SensorDataImport.dataset import dataset
from SensorDataImport.header import header
from SensorDataImport.signalProcessing import signalProcessing 
import copy
import os

leftFootFileName = "NRF52-92"
rightFootFileName = "NRF52-84"


def getFileNames(folderPath, fileEnding):
    fileNames = list()
    for file in os.listdir(folderPath):
        if file.endswith(fileEnding):
            print(os.path.join(folderPath, file))
            fileNames.append(os.path.join(folderPath, file))
    return fileNames;

def getFilesNamesPerFoot(path):
    leftFootPath = "";
    rightFootPath = "";
    for file in os.listdir(path):
        if file.endswith(".bin"):
            fullFileName = os.path.join(path, file);
            if leftFootFileName in fullFileName: 
                leftFootPath = fullFileName;
            if rightFootFileName in fullFileName:
                rightFootPath = fullFileName;
    return [leftFootPath, rightFootPath]

class session:
    leftFoot = None;
    rightFoot = None;

    def __init__(self, leftFoot, rightFoot):
        self.leftFoot = leftFoot
        self.rightFoot = rightFoot    
        
    @classmethod
    def from_filePaths(cls, leftFootPath, rightFootPath,header):
        session = cls(dataset(leftFootPath,header),dataset(rightFootPath,header))
        return session;
    
    @classmethod
    def from_folderPath(cls, folderPath, header):
        [leftFootPath, rightFootPath] = getFilesNamesPerFoot(folderPath);
        session = cls(dataset(leftFootPath,header),dataset(rightFootPath,header))
        return session;          
            
    def synchronizeFallback(self):   
        #cut away all sample at the beginning until both data streams are synchronized (SLAVE)
        inSync = (np.argwhere(self.leftFoot.sync > 0)[0])[0]
        self.leftFoot = self.leftFoot.cutDataset(inSync, len(self.leftFoot.counter));
        
        
        #cut away all sample at the beginning until both data streams are synchronized (MASTER)
        inSync = (np.argwhere(self.rightFoot.counter >= self.leftFoot.counter[0])[0])[0]
        self.rightFoot = self.rightFoot.cutDataset(inSync, len(self.rightFoot.counter));

        #cut both streams to the same lenght
        if(len(self.rightFoot.counter) >= len(self.leftFoot.counter)):
             length = len(self.leftFoot.counter)-1   
        else:
             length = len(self.rightFoot.counter)-1
    
        self.leftFoot = self.leftFoot.cutDataset(0, length);
        self.rightFoot = self.rightFoot.cutDataset(0, length);
        
    def synchronize(self):
        if(self.leftFoot.sessionHeader.syncRole == 'disabled' or self.rightFoot.sessionHeader.syncRole == 'disabled'):
            print("No header information found using fallback sync")
            self.synchronizeFallback();
            return
        try:
            if(self.leftFoot.sessionHeader.syncRole == self.rightFoot.sessionHeader.syncRole):
                print("ERROR: no master/slave pair found - synchronization FAILED")
                return;
        
            if(self.rightFoot.sessionHeader.syncRole == 'master'):
                master = self.rightFoot;
                slave = self.leftFoot;
            else:
                master = self.leftFoot;
                slave = self.rightFoot;
            
            try:
                inSync = (np.argwhere(slave.sync > 0)[0])[0]
            except:
                print("No Synchronization signal found - synchronization FAILED")
                return;
            
            #cut away all sample at the beginning until both data streams are synchronized (SLAVE)
            inSync = (np.argwhere(slave.sync > 0)[0])[0]
            slave = slave.cutDataset(inSync, len(slave.counter));
                
            #cut away all sample at the beginning until both data streams are synchronized (MASTER)
            inSync = (np.argwhere(master.counter >= slave.counter[0])[0])[0]
            master = master.cutDataset(inSync, len(master.counter));
    
            #cut both streams to the same lenght
            if(len(master.counter) >= len(slave.counter)):
                length = len(slave.counter)-1   
            else:
                length = len(master.counter)-1
        
            slave = slave.cutDataset(0, length);
            master = master.cutDataset(0, length);
        
            if(self.rightFoot.sessionHeader.syncRole == 'master'):
                self.rightFoot = master;
                self.leftFoot = slave;
            else:
                self.rightFoot = slave;
                self.leftFoot = master;
            #check if synchronization is valid
            #test synchronization
            deltaCounter = abs(self.leftFoot.counter - self.rightFoot.counter);
            sumDelta = np.sum(deltaCounter)
            if sumDelta != 0.0:
                print("ATTENTION: Error in synchronization. Check Data!")
        except Exception as e: 
            print(e)
            print("synchronization failed with ERROR")
        
    def calibrate(self):
        self.leftFoot.calibrate();
        self.rightFoot.calibrate();
    
    def rotateAxis(self,system):
        if(system == 'egait'):
            self.leftFoot.rotateAxis('gyro',2,0,1,-1,-1,1) # swap axis Z,X,Y, change sign -X-Y+Z
            self.leftFoot.rotateAxis('acc',2,0,1,-1,-1,1)            
            self.rightFoot.rotateAxis('gyro',2,0,1,1,1,-1)
            self.rightFoot.rotateAxis('acc',2,0,1,1,-1,-1)
            self.leftFoot.rotateAxis('pressure',0,0,0,0,0,0);
            self.rightFoot.rotateAxis('pressure',0,0,0,0,0,0);
        elif(system ==  'default'):
            self.rightFoot.rotateAxis('default',0,0,0,0,0,0);
            self.leftFoot.rotateAxis('default',0,0,0,0,0,0);
        else:
            print('unknown system, you need to handle axis rotation per foot yourself!')
    
    def cutData(self, start,stop):
        session = copy.copy(self);
        session.leftFoot = session.leftFoot.cutDataset(start,stop);
        session.rightFoot = session.rightFoot.cutDataset(start,stop);
        return session;
    
    
        
        