"""Packages"""
import serial
import matplotlib.pyplot as plt
import time
#import numpy as np
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
"""Plot data in realtime"""

import matplotlib.pyplot as plt
import numpy as np

s_sensor = serial.Serial(port="COM8", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_printer = serial.Serial(port="COM10", baudrate=250000)
s_Force = serial.Serial(port = "COM11", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
feedrate = "1000"

#setpos(1,1,0, s_printer)
initialize_printer(s_printer)
#time.sleep(1)

"""Load Normalization Values of Model"""
model_name = "_AFG_board2_screw_2"
norm_val_og= np.loadtxt("./Data/norm_val_"+model_name+".txt",dtype = float)
b15_norm = []
print("Collecting Norm Val")
for i in range(250):
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
print("Training Norm Values: ",norm_val_og)
print(f"New Norm Values: ", norm_val)
#norm_val = norm_val_og

b = read_sensor(s_sensor)
b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
print("Sensor Values: ",b)#Check if sensor works
print("Normalized Sensor Values: ",b15 /norm_val)#Check if sensor works
print("New Norm Values: ",b15 /norm_val_og)#Check if sensor works

#calibrate_force(s_printer,s_Force)


"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU, output_hidden=[200,200],
                            output_activation=nn.ReLU)

model.load_state_dict(torch.load("./Data/MLP_"+model_name))
print(model.eval)
truths = [0,0,0]



"""Setup Plot"""
i = 0
show = 5 #Ammount of datapoints to be shown simultaneously
x_list = np.zeros(show)
y_list = np.zeros(show)
F_list = np.zeros(show)
x_truths = np.zeros(show)
y_truths = np.zeros(show)
F_truths = np.zeros(show)
truths = [0,0,0]

plt.ion()

fig = plt.figure()
ax = fig.add_subplot(111)
scatter = ax.scatter(x_list,y_list, marker = ".",c="b",s=F_list)
ax.set_xlim(0,20)
ax.set_ylim(0,20)
plt.draw()


for samples in range(5):
    print("---------------")
    loc = np.random.randint(5, 15, 2)
    depths = np.random.randint(25, 32, 1)
    depths = round(depths[0] / 10, 1)
    setpos(loc[0], loc[1], 0, s_printer)
    setpos(loc[0], loc[1], -depths, s_printer)
    time.sleep(0.5)
    b = read_sensor(s_sensor)
    truth = [loc[0], loc[1], read_force(s_Force)]
    b15 = np.array(np.concatenate((b[0:3], b[4:7], b[8:11], b[12:15], b[16:19])))
    b15 = b15 / norm_val
    # print(b15)
    single_set = [torch.tensor(b15, dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    xyF = model(single_set[0])
    xyF = xyF.detach().numpy()
    xyF = [round(a, 4) for a in xyF]
    print(f"Predicted [X, Y, F] {xyF}")
    print(f"Real      [X, Y, F] {truth}, (Z: {read_printer(s_printer)[2]})")



    """Plot Data"""
    if True:
        x_list[i] = xyF[0]
        y_list[i] = xyF[1]
        F_list[i] = xyF[2]

        x_truths[i] = truth[0]
        y_truths[i] = truth[1]
        F_truths[i] = truth[2]

    i += 1
    if i == show - 1:
        i = 0
    scatter = ax.scatter(x_list,y_list, marker = ".",c="b",s=F_list)
    print(x_list)
    plt.draw()
    plt.pause(0.25)

    time.sleep(1)
    setpos(loc[0], loc[1], 0, s_printer)
    setpos(0, 0, 0, s_printer)

plt.ioff()
plt.show()
"""
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