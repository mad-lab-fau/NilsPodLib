# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from NilsPodLib import session as sensor

plt.close('all')

import ntpath
import tkinter as tk
from tkinter import filedialog


def path_leaf(path):
    head, tail = ntpath.split(path)
    return tail or ntpath.basename(head)

root = tk.Tk()
root.withdraw()

file_path_leftFoot = filedialog.askopenfilename()
file_name_leftFoot = path_leaf(file_path_leftFoot)


root = tk.Tk()
root.withdraw()

file_path_rightFoot = filedialog.askopenfilename()
file_name_rightFoot = path_leaf(file_path_rightFoot)

#since firmware verison V0.2.0 a header is included within each binary data file (=> header flag has to be enabled/disabled accordingly)
header = 1;
freeRTOS = 1
print("Reading in Data...")
#session = sensor.session.from_folderPath(folder_path,header);
#session = sensor.session.from_filePaths(file_path_leftFoot,file_path_rightFoot,header);
session = sensor.session(sensor.dataset(file_path_leftFoot,header,freeRTOS),sensor.dataset(file_path_rightFoot,header,freeRTOS))
print("Data Sucessfully Loaded")


#session.calibrate();
#session.rotateAxis('egait');

#session.leftFoot = session.leftFoot.interpolateDataset(session.leftFoot);

session.synchronize();

plt.figure();
plt.plot(session.leftFoot.counter,'r')
plt.plot(session.rightFoot.counter,'b')
plt.plot(session.leftFoot.counter - session.rightFoot.counter)


plt.figure();
plt.plot(session.leftFoot.gyro.data[:,2],'r')
plt.plot(session.rightFoot.gyro.data[:,2],'b')


fig, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(session.leftFoot.gyro.data[:,2], 'b')
axarr[1].plot(session.rightFoot.gyro.data[:,2],'r')

fig, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(session.leftFoot.gyro.data[:,2])
axarr[0].plot(session.rightFoot.gyro.data[:,2])
axarr[1].plot(session.leftFoot.pressure.data)
axarr[1].plot(session.rightFoot.pressure.data)

fig, axarr = plt.subplots(2, sharex=True)
axarr[0].plot(session.leftFoot.pressure.data)
axarr[1].plot(session.rightFoot.pressure.data)

fig, axarr = plt.subplots(2, sharex=True,sharey=True)
axarr[0].plot(session.leftFoot.acc.data)
axarr[1].plot(session.rightFoot.acc.data)

fig, axarr = plt.subplots(3, sharex=True)
axarr[0].plot(session.leftFoot.pressure.data)
axarr[1].plot(session.leftFoot.gyro.data[:,2])
axarr[2].plot(session.leftFoot.acc.data)

fig = plt.figure()
plt.plot(session.leftFoot.acc.data)
plt.plot(session.rightFoot.acc.data)


fig, ax1 = plt.subplots()
ax1.plot(session.leftFoot.counter)
ax1.plot(session.rightFoot.counter)
ax2 = ax1.twinx()
ax2.plot(session.leftFoot.sync)
ax2.plot(session.rightFoot.sync)
plt.plot(abs((1.0*session.rightFoot.counter)-session.leftFoot.counter), color='r')

# Four axes, returned as a 2-d array
f, axarr = plt.subplots(2, 2,sharex=True)
axarr[0, 0].plot(session.leftFoot.gyro.data[:,2])
axarr[0, 0].plot(session.rightFoot.gyro.data[:,2])

#axarr[0, 0].set_title('Axis [0,0]')
axarr[0, 1].plot(session.leftFoot.pressure.data)
#axarr[0, 1].set_title('Axis [0,1]')
axarr[1, 0].plot(session.leftFoot.acc.data)
#axarr[1, 0].set_title('Axis [1,0]')
axarr[1, 1].plot(session.leftFoot.baro.data)
axarr[1, 1].plot(session.rightFoot.baro.data)
#axarr[1, 1].set_title('Axis [1,1]')


session.rightFoot.exportCSV(file_name_rightFoot + "_raw.csv");
session.leftFoot.exportCSV(file_name_leftFoot + "_raw.csv");