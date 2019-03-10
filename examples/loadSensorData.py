# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
from NilsPodLib import session as sensor
import numpy as np
import tkinter as tk
from tkinter import filedialog

plt.close('all')

root = tk.Tk()
root.withdraw()
file_path = filedialog.askopenfilename()

dataset = sensor.dataset(file_path)
# dataset.calibrate();

seconds = dataset.header.unixTime_stop - dataset.header.unixTime_start
n = len(dataset.counter)
if seconds > 0:
    print("Start: " + str(dataset.header.convertUnixTimeToDateTime(dataset.header.unixTime_start)))
    print("Stop: " + str(dataset.header.convertUnixTimeToDateTime(dataset.header.unixTime_stop)))
    print("Sampling Frequency calculated: " + str(1 / (seconds / n)) + "Hz")
else:
    print("Timestamp Error")

if dataset.header.batteryEnabled:
    plt.figure()
    ax1 = plt.plot(dataset.battery.data)
    plt.ylim(0, 5)
    plt.title('Battery')

if dataset.header.baroEnabled:
    plt.figure()
    plt.plot(dataset.baro.data)
    plt.title('Baro')

if dataset.header.pressureEnabled:
    plt.figure()
    plt.plot(dataset.pressure.data)
    plt.title('Pressure')

plt.figure()
plt.plot(dataset.counter)
# plt.plot(dataset.sync*np.max(dataset.counter))
plt.title('Counter')

plt.figure()
plt.plot(dataset.acc.data)
plt.title('Accelerometer')

plt.figure()
plt.plot(dataset.gyro.data)
plt.title('Gyroscope')
