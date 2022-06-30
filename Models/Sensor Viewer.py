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

import matplotlib
matplotlib.use("TkAgg")

sys.path.append("./Functions")
from Model_functions import *
from SensorCollectionFunctions import *

s_sensor = serial.Serial(port="COM3", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
"""Load Normalization Values of Model"""
model_name = "_AFG_board2"
norm_val_og= np.loadtxt("./Data/norm_val_"+model_name+".txt",dtype = float)
b15_norm = []
print("Collecting Norm Val")
for i in range(10):
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
print("Training Norm Values: ", norm_val_og)
print(f"New Norm Values: ", norm_val)
#norm_val = norm_val_og

b = read_sensor(s_sensor)
b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
print("Sensor Values: ",b)#Check if sensor works
print("Normalized Sensor Values: ",b15 /norm_val)#Check if sensor works
print("New Norm Values: ",b15 /norm_val_og)#Check if sensor works

"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU, output_hidden=[200,200],
                            output_activation=nn.ReLU)

model.load_state_dict(torch.load("./Data/MLP_"+model_name))
print(model.eval)
truths = [0,0,0]

for i in range(100):
    print("---------------")
    time.sleep(0.5)
    b = read_sensor(s_sensor)
    b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
    b15 = b15 / norm_val
    #print(b15)
    single_set = [torch.tensor(b15, dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    xyF = model(single_set[0])
    xyF = xyF.detach().numpy()
    xyF = [round(a,4) for a in xyF]
    print(f"Predicted [X, Y, F] {xyF}")
    time.sleep(0.5)





"""Plot data in realtime"""
"""
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