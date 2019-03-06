# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from SensorDataImport import session as sensor
import datetime as datetime

plt.close('all')


folder_path = '/Users/nils/Desktop/'
#file_path_leftFoot = '/Users/nils/Desktop/TestData/InsoleLeft-38_TBD-Recovery.bin'
#file_path_rightFoot = '/Users/nils/Desktop/TestData/InsoleRight-38_22_06_2018-12-29-18.bin'

#file_path_rightFoot = '/Users/nils/Documents/Lehrstuhlaufgaben/Student_Thesis/Students/2018_BA_Philipp_Joneck/Study/Data/Sub004-38/InsoleRight-38_13_06_2018-11-27-18.bin'
#file_path_leftFoot = '/Users/nils/Documents/Lehrstuhlaufgaben/Student_Thesis/Students/2018_BA_Philipp_Joneck/Study/Data/Sub004-38/InsoleLeft-38_13_06_2018-11-27-18.bin'

file_path_rightFoot = '/Users/nils/Desktop/HomeMonitoringData/Sophia/InsoleRight-38_28_06_2018-07-24-29.bin'
file_path_leftFoot = '/Users/nils/Desktop/HomeMonitoringData/Sophia/InsoleLeft-38_28_06_2018-07-24-29.bin'
file_path_labels = '/Users/nils/Desktop/HomeMonitoringData/Sophia/log_2018-06-28.csv'


file_path_rightFoot = '/Users/nils/Desktop/HomeMonitoringData/Nils/23_06_2018_Nils/InsoleRight-42_23_06_2018-11-42-25.bin'
file_path_leftFoot = '/Users/nils/Desktop/HomeMonitoringData/Nils/23_06_2018_Nils/InsoleLeft-42_23_06_2018-11-42-25.bin'
file_path_labels = '/Users/nils/Desktop/HomeMonitoringData/Nils/23_06_2018_Nils/log_2018-06-23.csv'


file_path_rightFoot = '/Users/nils/Desktop/HomeMonitoringData/Leonie/25_06_2018_Leonie/InsoleRight-38_25_06_2018-16-47-37.bin'
file_path_leftFoot = '/Users/nils/Desktop/HomeMonitoringData/Leonie/25_06_2018_Leonie/InsoleLeft-38_25_06_2018-16-47-37.bin'
file_path_labels = '/Users/nils/Desktop/HomeMonitoringData/Leonie/25_06_2018_Leonie/log_2018-06-25.csv'



#since firmware verison V0.2.0 a header is included within each binary data file (=> header flag has to be enabled/disabled accordingly)
header = 1;
print("Reading in Data...")
session = sensor.session(sensor.dataset(file_path_leftFoot,header),sensor.dataset(file_path_rightFoot,header))
print("Data Sucessfully Loaded")
session.calibrate();
session.rotateAxis('egait');


lLeft = len(session.leftFoot.counter)
lRight = len(session.rightFoot.counter)

[a,b] = session.synchronize();


startTimeUnixLeft = session.leftFoot.sessionHeader.unixTime_start + (a/200.0)
stopTimeUnixLeft = session.leftFoot.sessionHeader.unixTime_stop
startTimeUnixRight = session.rightFoot.sessionHeader.unixTime_start + (b/200.0)
stopTimeUnixRight = session.rightFoot.sessionHeader.unixTime_stop


startTimeUnix = (startTimeUnixLeft + startTimeUnixRight)/2.0
stopTimeUnix = (stopTimeUnixLeft + stopTimeUnixRight)/2.0
startTime = session.leftFoot.sessionHeader.convertUnixTimeToDateTime(startTimeUnix)
stopTime = session.leftFoot.sessionHeader.convertUnixTimeToDateTime(stopTimeUnix)

#startTime = session.leftFoot.sessionHeader.timeStamp;
#stopTime = session.leftFoot.sessionHeader.stop_timeStamp;


date = startTime.date();

ary = np.genfromtxt(file_path_labels,delimiter=',',dtype=str)

labelTimeList = []
labelList = []




fig, axarr = plt.subplots(1, sharex=True,sharey=True, figsize=(12,6))
plt.plot(session.leftFoot.gyro.data[:,2]);
plt.plot(session.rightFoot.gyro.data[:,2]);


labelOffset = 3000;

for i in range(1,len(ary)):    
    labelName = ary[i,0];
    labelList.append(labelName);
    time = datetime.datetime.strptime(ary[i,1], '%H:%M:%S').time()
    dtime = datetime.datetime.combine(date,time);
    dt = dtime - startTime;
    dt_sec = dt.seconds
    labelTimeList.append(dt_sec)
    sampleIdx = (dt_sec*200)+labelOffset
    plt.axvline(x=sampleIdx, c = 'r')
    plt.text(sampleIdx,0,labelName,rotation = 90)
    
print(len(labelList))


fig, axarr = plt.subplots(1, sharex=True,sharey=True, figsize=(12,6))
plt.plot(session.leftFoot.gyro.data[:,2]);
plt.plot(session.rightFoot.gyro.data[:,2]);
  

for i in range(0,len(labelList)):    
    labelName = labelList[i];
    labelTimeList.append(dt_sec)
    sampleIdx = (labelTimeList[i]*200)+labelOffset
    plt.axvline(x=sampleIdx, c = 'r')
    plt.text(sampleIdx,0,labelName,rotation = 90)
    
print(len(labelList))



for i in range(0, len(labelList)):
    #plt.close('all')
    sampleIdx = (labelTimeList[i]*200)+labelOffset
    start = sampleIdx - 6000;
    stop = sampleIdx + 6000;
    if(start < 0):
        start = 0;
    if(i > 0):
        plt.axvline(x=(labelTimeList[i-1]*200)+labelOffset, c = 'r')
    plt.axvline(x=sampleIdx, c = 'g')
    plt.xlim([start,stop])
    plt.pause(0.1)
    plt.show()
    x = input();
    if(x == 'q'):
        break;


plotPlotly = 0

if plotPlotly:
    gyrLeft = session.leftFoot.gyro.data[:,2];
    
    import plotly.plotly as py
    import plotly.graph_objs as go
    from plotly import __version__
    from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplot
    
    # Create random data with numpy
    import numpy as np
    
    random_x = np.linspace(0, len(gyrLeft), len(gyrLeft))
    
    
    plot([go.Scattergl(x=random_x, y=gyrLeft)])



    
