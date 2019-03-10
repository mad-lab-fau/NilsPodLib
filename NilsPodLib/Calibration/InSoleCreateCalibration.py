#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:01:33 2017

@author: nils
"""

import pandas as pd
import matplotlib.pyplot as plt

import numpy as np
from DataHandling.InSoleLoadBinary import InSoleLoadBinary
from DataHandling.loadLightBlueCSV import loadLightBlueCSV, loadEgaitData
from mpl_toolkits.mplot3d import Axes3D
import scipy.io
from scipy.optimize import minimize, leastsq
import pickle

plt.close('all')


def cost_function(y, a):
    T = np.matrix([[1, -y[0], -y[1]], [0, 1, -y[2]], [0, 0, 1]])
    K = np.matrix([[y[3], 0, 0], [0, y[4], 0], [0, 0, y[5]]])
    b = np.matrix([y[6], y[7], y[8]]).T
    A = T * K * (a - b)
    return 1 - np.apply_along_axis(np.linalg.norm, 0, A)


def drawSphere(xCenter, yCenter, zCenter, r):
    # draw sphere
    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)
    # shift and scale sphere
    x = r * x + xCenter
    y = r * y + yCenter
    z = r * z + zCenter
    return x, y, z


def running_mean(x, N):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / N


def rolling_window(a, window):
    shape = a.shape[:-1] + (a.shape[-1] - window + 1, window)
    strides = a.strides + (a.strides[-1],)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def windowed_variance(data, i, window):
    return np.var(data[i - (window / 2):i + (window / 2)])


def get_static_intervals(accData, gyrData, windowSize, thres):
    accNorm = np.empty(len(accData))
    accVar = np.empty(len(accData))
    accVarX = np.empty(len(accData))
    accVarY = np.empty(len(accData))
    accVarZ = np.empty(len(accData))

    gyrVarAxes = np.empty((len(gyrData), 3,))
    gyrVar = np.empty(len(accData))
    detector = np.empty(len(accData))
    gyrDetector = np.empty(len(accData))

    for i in range(0, len(accData)):
        accNorm[i] = np.linalg.norm(accData[i, :])

    # for i in range(0,len(accData)):
    #    vector[i] = np.insert(accData[i,:],0,[0,0,0]);

    for i in range(windowSize, len(accData) - windowSize):
        accVarX[i] = windowed_variance(accData[:, 0], i, windowSize)
        accVarY[i] = windowed_variance(accData[:, 1], i, windowSize)
        accVarZ[i] = windowed_variance(accData[:, 2], i, windowSize)
        accVar[i] = np.sqrt(accVarX[i] ** 2 + accVarY[i] ** 2 + accVarZ[i] ** 2)
        if accVar[i] < thres:
            detector[i] = 2000
        else:
            detector[i] = 0

    for i in range(windowSize, len(gyrData) - windowSize):
        gyrVarAxes[i, 0] = windowed_variance(gyrData[:, 0], i, windowSize)
        gyrVarAxes[i, 1] = windowed_variance(gyrData[:, 1], i, windowSize)
        gyrVarAxes[i, 2] = windowed_variance(gyrData[:, 2], i, windowSize)
        gyrVar[i] = np.sqrt(gyrVarAxes[i, 0] ** 2 + gyrVarAxes[i, 1] ** 2 + gyrVarAxes[i, 2] ** 2)
        if gyrVar[i] < 150:
            gyrDetector[i] = 2000
        else:
            gyrDetector[i] = 0

    for i in range(0, 400):
        detector[i] = 0
        detector[len(detector) - i - 1] = 0

    return detector


# filename = "NRF52-84965C0_25_08_2017-01-39-16.bin"
# filename = "NRF52-24CAB1AB_26_08_2017-15-33-29.bin"
sensor = 'NRF52-92338C81'
filename = sensor + "_Calibration.bin"
path = 'Calibration/CalibrationSessions/' + filename

checkCounter = 0
plot = True

if path.endswith(".bin"):
    [accData, gyrData, pressure, force, battery, counter, sync] = InSoleLoadBinary(path, 0)
else:
    [accData, gyrData, pressure, force, battery, counter, sync] = loadLightBlueCSV(path)

if plot:
    plt.figure()
    plt.plot(accData)

detector = get_static_intervals(accData, gyrData, 300, 150)

start = 0
n = 0
g = 0
sumX = 0
sumY = 0
sumZ = 0
vector = np.zeros((len(accData), 6,))
gyr = [0, 0, 0]

for i in range(1, len(accData)):
    if (detector[i] > 0) and (detector[i - 1] == 0):
        start = i
    if start > 0:
        gyr += gyrData[i, :]
        g = g + 1
        sumX = sumX + accData[i, 0]
        sumY = sumY + accData[i, 1]
        sumZ = sumZ + accData[i, 2]
    if (detector[i] == 0) and (detector[i - 1] > 0):
        if i - start > 200:
            vector[n, 3] = sumX / (i - start)
            vector[n, 4] = sumY / (i - start)
            vector[n, 5] = sumZ / (i - start)
            n = n + 1
        sumX = 0
        sumY = 0
        sumZ = 0
        start = 0

gyr = gyr / g

vector = vector[0:n, :]

if plot:
    plt.figure()
    plt.plot(accData)
    plt.plot(detector)

scale_acc = 2048.0
x0 = [0.0, 0.0, 0.0, 1.0 / scale_acc, 1.0 / scale_acc, 1.0 / scale_acc, 0.0, 0.0, 0.0]
static_acceleration_vectors = vector[:, 3:6].T

xn = leastsq(cost_function, x0, args=static_acceleration_vectors)[0]

Ta = np.matrix([[1.0, -xn[0], -xn[1]], [0.0, 1.0, -xn[2]], [0.0, 0.0, 1.0]])
Ka = np.matrix([[xn[3], 0.0, 0.0], [0.0, xn[4], 0.0], [0.0, 0.0, xn[5]]])
ba = np.matrix([xn[6], xn[7], xn[8]]).T

scale_gyro = 16.4
Tg = np.matrix([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
Kg = np.matrix([[1.0 / scale_gyro, 0.0, 0.0], [0.0, 1.0 / scale_gyro, 0.0], [0.0, 0.0, 1.0 / scale_gyro]])
bg = np.matrix(gyr).T

print('Ta:')
print(Ta)
print('Ka:')
print(Ka)
print('ba:')
print(ba)
print('Tg:')
print(Tg)
print('Kg:')
print(Kg)
print('bg:')
print(bg)

with open(sensor + '.pickle', 'w') as f:  # Python 3: open(..., 'wb')
    pickle.dump([Ta, Ka, ba, Tg, Kg, bg], f)

if plot:
    accDataCalib = (Ka * Ta * (accData.T - ba)).T
    gyroDataCalib = (Kg * Tg * (gyrData.T - bg)).T

    plt.figure()
    plt.plot(accData * (1 / scale_acc))
    plt.plot(accDataCalib)
    plt.figure()
    plt.plot(gyrData * (1 / scale_gyro))
    plt.plot(gyroDataCalib)

    X, Y, Z, U, V, W = zip(*vector)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.quiver(X, Y, Z, U, V, W, color="r", lw=0.5)
    ax.set_xlim([-2200, 2200])
    ax.set_ylim([-2200, 2200])
    ax.set_zlim([-2200, 2200])
    (xs, ys, zs) = drawSphere(0, 0, 0, 2048)
    ax.plot_wireframe(xs, ys, zs, color="b", lw=0.5, alpha=.5)
    ax.scatter(static_acceleration_vectors[0, :], static_acceleration_vectors[1, :], static_acceleration_vectors[2, :],
               '.', s=0.5)
    plt.show()

    static_acceleration_vectors_calib = (Ta * Ka * (static_acceleration_vectors - ba))

    fig = plt.figure('cubic')
    ax = fig.add_subplot(221, projection='3d')
    ax.set_xlim([-1.2, 1.2])
    ax.set_ylim([-1.2, 1.2])
    ax.set_zlim([-1.2, 1.2])
    (xs, ys, zs) = drawSphere(0, 0, 0, 1)
    ax.plot_wireframe(xs, ys, zs, color="b", lw=0.5, alpha=.5)
    ax.scatter([static_acceleration_vectors_calib[0, :]], [static_acceleration_vectors_calib[1, :]],
               [static_acceleration_vectors_calib[2, :]], '.', s=0.5, color='r')
    # ax.scatter(static_acceleration_vectors_calib[0,:],static_acceleration_vectors_calib[1,:],static_acceleration_vectors_calib[2,:],'.',s=0.5)

    ax = fig.add_subplot(222)
    ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
    ax.scatter([static_acceleration_vectors_calib[0, :]], [static_acceleration_vectors_calib[1, :]], s=0.5, color='r')

    ax = fig.add_subplot(223)
    ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
    ax.scatter([static_acceleration_vectors_calib[0, :]], [static_acceleration_vectors_calib[2, :]], s=0.5, color='r')

    ax = fig.add_subplot(224)
    ax.add_artist(plt.Circle((0, 0), 1, color='b', fill=False))
    ax.scatter([static_acceleration_vectors_calib[1, :]], [static_acceleration_vectors_calib[2, :]], s=0.5, color='r')

    plt.show()
