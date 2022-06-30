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

s_sensor = serial.Serial(port="COM8", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_printer = serial.Serial(port="COM10", baudrate=250000)
s_Force = serial.Serial(port = "COM11", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
feedrate = "1000"

setpos(1,1,0, s_printer)
initialize_printer(s_printer)
#time.sleep(1)

"""Load Normalization Values of Model"""
model_name = "_AFG_Board1_50"
norm_val_og= np.loadtxt("./Data/norm_val_"+model_name+".txt",dtype = float)
b15_norm = []
print("Collecting Norm Val")
for i in range(1000):
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

norm_val = norm_val_og
"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU, output_hidden=[200,200],
                            output_activation=nn.ReLU)

model.load_state_dict(torch.load("./Data/MLP_"+model_name))
print(model.eval)
truths = [0,0,0]
xyF_list = []
truths_list = []
for i in range(9):
    print("---------------")
    loc = np.random.randint(5,15,2)
    depths = np.random.randint(25,32,1)
    depths = round(depths[0]/10,1)
    setpos(loc[0],loc[1], 0 ,s_printer)
    setpos(loc[0], loc[1], -depths, s_printer)
    time.sleep(0.5)
    b = read_sensor(s_sensor)
    truth = [loc[0], loc[1], read_force(s_Force)]
    b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
    b15 = b15 / norm_val
    #print(b15)
    single_set = [torch.tensor(b15, dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    xyF = model(single_set[0])
    xyF = xyF.detach().numpy()
    xyF = [round(a,4) for a in xyF]
    print(f"Predicted [X, Y, F] {xyF}")
    print(f"Real      [X, Y, F] {truth}, (Z: {read_printer(s_printer)[2]})")
    time.sleep(3)
    setpos(loc[0], loc[1], 0, s_printer)
    setpos(0, 0, 0, s_printer)
    time.sleep(0.5)
    xyF_list.append(xyF)
    truths_list.append(truth)

setpos(1,1,0,s_printer)
print('Sending: ' + "G92")
s_printer.write("G92 X0 Y0 Z0\n".encode())


colors = ["r", "g", "b", "y", "m", "c", "k", "burlywood"]
colors = ["#FF00FF", "#009933", "#0000FF", "#CC3300", "#99FF33", "#00FFFF", "#663300", "#FFFF00"]
for i,output in enumerate(xyF_list[1:]):
    plt.plot(output[0], output[1], "o",
                     markersize=output[2]*10+5, markerfacecolor='none', markeredgecolor=colors[i])
    label = truths_list[i]
    plt.plot(label[0], label[1], "x", label=str(label[2]) + "N == " + str(output[2]) + "N",
             markersize=(label[2]) * 10+1, color=colors[i])
    print(output,label)
plt.xlim(0,20)
plt.ylim(0,20)
plt.legend()
plt.title("Sensor Test Results ", model_name)
plt.show()