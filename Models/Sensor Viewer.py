"""Packages"""
import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime
import torch
import torch.nn as nn

sys.path.append("./Functions")

from SensorCollectionFunctions import *
from Model_functions import *



sys.path.append("./Functions")
from Model_functions import *
from SensorCollectionFunctions import *


"""Setup"""
port = "COM8" #Set port Arduino with sensor is on. (5X_Burst_stream.ino)
model_name = "_AFG_board2"# _at the start, copy from moddel trainer
new_normal = "current" #current/training, for new normalization values/at training values, def = "current"
testing = 0 # True/False Prints everything. and I mean EVERYTHING
delay = 0.5 #Time between printing new prediction, default = 0.5s
iterations = 15 #How many times the program should predict
plot_at_end = 1 #True/False, Wether to plot the results at the end


"""Set port for sensor"""
s_sensor = serial.Serial(port=port, baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

"""Load Normalization Values of Model"""
norm_val_og= np.loadtxt("./Data/norm_val_"+model_name+".txt",dtype = float)
b15_norm = []
print("Collecting Norm Val, this may take a second")
if new_normal == "current":
    for i in range(15):
        b = read_sensor(s_sensor)
        b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
        b15_norm.append(b15)
    norm_val = []
    for i in range(len(b15_norm[0])):
        mean = 0
        for count, b in enumerate(b15_norm):
            mean += b[i]
        mean = mean / count
        norm_val.append(mean)
    if testing:
        print(f"New Norm Values: ", [round(a,2) for a in norm_val])
else:
    norm_val = norm_val_og
if testing:
    print("Training Norm Values: ", [round(a,2) for a in norm_val_og])

#Check if sensor works
b = read_sensor(s_sensor)
b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
print("\nSensor Values: ",b)
print("Normalized Sensor Values: ",np.round(b15 /norm_val,2))#Check if sensor works
norm_val = np.array(norm_val)

if new_normal == "current" and testing:
    dev = norm_val-norm_val_og
    print("Deviation from original norm:", [round(a,0) for a in dev])
    print("Deviation %:", [str(int(round(a*100,0)))+"%" for a in dev/norm_val])

"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU, output_hidden=[200,200],
                            output_activation=nn.ReLU)#Setup model, carefull to have same pararms as during training

if testing:
    print()
    print(model.eval)

model.load_state_dict(torch.load("./Data/MLP_"+model_name))#Load pretrained model
truths = [0,0,0]

print("\n----------------------------------------")
print("Predicted [X, Y, F]: [mm, mm, N]")
xyF_list = []

for i in range(iterations):
    print("----------------------------------------")
    time.sleep(0.5)
    b = read_sensor(s_sensor)# Collects 20 datapoints from sensor, x,y,f,T, from 5 sensors
    b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))# We only want x,y,f
    b15 = b15 / norm_val #Normalize sensor values
    if testing:
        print("b15: ", [round(a,2) for a in b15])
    single_set = [torch.tensor(b15, dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    #Convert to proper shape for pytorch
    #TODO Make pytorch nograd so we dont need this weird conversion
    xyF = model(single_set[0])
    xyF = xyF.detach().numpy()#Turn to array
    xyF = [round(a,2) for a in xyF]
    xyF_list.append(xyF)
    if xyF[2] < 0.02 and not testing: #Predicts not being touched as 0N (0.02 since AFG LOD is 0.02)
        xyF = "Sensor not being touched"
    print(f"Predicted [X, Y, F]: {xyF}")
    time.sleep(delay)

if plot_at_end:
    for i, output in enumerate(xyF_list[1:]):
        plt.plot(output[0], output[1], "o",
                 markersize=output[2] * 10)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.title("Sensor Test Results " + model_name)
    plt.show()


"""Plot data in realtime"""
"""
def plotInRT(xyF):
    updated_x = 1
    updated_y = 2
    Force = xyF[2]

    line1.set_xdata(updated_x)
    line1.set_ydata(updated_y)
    plt.legend(f"Force = {F} N")
    print("UX",updated_x)
    figure.canvas.draw()

    figure.canvas.flush_events()
    time.sleep(0.1)

if plot_in_RT:
    x = 0
    y = 0
    F = 0

    plt.ion()

    figure, ax = plt.subplots(figsize=(8, 6))
    line1, = ax.plot(x, y)

    plt.title("Dynamic Plot of sensor readings", fontsize=25)

    plt.xlabel("X [mm]", fontsize=18)
    plt.ylabel("Y [mm]", fontsize=18)
    plt.xlim(0, 20)
    plt.ylim(0, 20)
    plt.legend(f"Force = {F} N")
    plt.title("Sensor Test Results " + model_name)
import matplotlib.pyplot as plt
import numpy as np
i = 0
show = 15 #Ammount of datapoints to be shown simultaneously
x_list = np.zeros(show)
y_list = np.zeros(show)
F_list = np.zeros(show)
truths = [0,0,0]

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x_list, y_list)
ax.set_xlim(0,20)
ax.set_ylim(0,20)
plt.draw()

for samples in range(0):
    single_set = [torch.tensor(read_sensor(), dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    xyF = model(single_set[0])
    # print(f"X:{float(xyF[0])}, Y:{float(xyF[1])}, F:{float(xyF[2])}")
    if (xyF[2] != 0):
        # print(float(xyF[0]),float(xyF[1]),float(xyF[2]))
        x_list[i] = float(xyF[0])+1*i
        y_list[i] = float(xyF[1])
        F_list[i] = float(xyF[2]) * 25
    else:
        # print(float(xyF[0]),float(xyF[1]),float(xyF[2]))
        x_list[i] = 1*i
        y_list[i] = 0
        F_list[i] = 0
    i += 1
    if i == show - 1:
        i = 0
    line1, = ax.plot(x_list, y_list)
    plt.draw()
    plt.pause(2)

plt.ioff()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

x = x_list
y = y_list

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
line1, = ax.plot(x, y, 'r-')
plt.draw()

for phase in np.linspace(0, 10*np.pi, 500):
    x_list[3] = x_list[3]+1
    line1.set_ydata(y_list)
    line1.set_xdata(x_list)
    plt.draw()
    plt.pause(0.02)

plt.ioff()
plt.show()
"""