# -*- coding: utf-8 -*-




import matplotlib.pyplot as plt
from SensorDataImport import session as sensor
import numpy as np
import tkinter as tk
from tkinter import filedialog

plt.close('all')


root = tk.Tk()
root.withdraw()

file_path = filedialog.askopenfilename()


header = 1;
plots = 1;
freeRTOS = 1;

#dataset = sensor.dataset(file_path,header,freeRTOS)
dataset = sensor.dataset(file_path)
#dataset.calibrate();
#dataset.rotateAxis();


seconds = dataset.header.unixTime_stop - dataset.header.unixTime_start
n = len(dataset.gyro.data)
if(seconds > 0):
    print("Start: " + str(dataset.header.convertUnixTimeToDateTime(dataset.header.unixTime_start)))
    print("Stop: " + str(dataset.header.convertUnixTimeToDateTime(dataset.header.unixTime_stop)))
    print("Sampling Frequency calculated: " + str(1/(seconds/n)) + "Hz")
else:
    print("Timestamp Error");

        



if plots:
    fig = plt.figure();
    ax1 = plt.plot(dataset.battery.data);
    plt.ylim(0, 5)
    plt.title('Battery')
    
    fig = plt.figure();
    plt.plot(dataset.baro.data);
    plt.title('Baro')
    
    fig = plt.figure();
    plt.plot(dataset.pressure.data);
    plt.title('Pressure')
    
    fig = plt.figure();
    plt.plot(dataset.acc.data);
    plt.title('Accelerometer')
    
    fig = plt.figure();
    plt.plot(dataset.gyro.data);
    plt.title('Gyroscope')
    
    fig = plt.figure();
    plt.plot(dataset.counter);
    plt.plot(dataset.sync*np.max(dataset.counter))
    plt.title('Counter')


