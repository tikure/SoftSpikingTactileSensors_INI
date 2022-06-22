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

s_printer = serial.Serial(port="COM4", baudrate=250000)
s_Force = serial.Serial(port = "COM7", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_sensor = serial.Serial(port="COM5", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

feedrate = "1600"
setpos(0,0,0, s_printer)
initialize_printer(s_printer)
time.sleep(1)

"""Load Normalization Values of Model"""
model_name = "AFG_test"
norm_val = np.loadtxt("./Data/norm_val__"+model_name+".txt",dtype = float)
print(f"Normalization Values: {len(norm_val)},{norm_val}")

print("Normalized Sensor Values: ",read_sensor(s_sensor))#Check if sensor works

"""Setup Model"""
model = vanilla_model(15, feature_dim=40, feat_hidden=[200,200], activation_fn=nn.ReLU, output_hidden=[200,200],
                            output_activation=nn.ReLU)

model.load_state_dict(torch.load("./Data/MLP__"+model_name))
print(model.eval)
truths = [0,0,0]
while 1:
    print("---------------")
    loc = np.random.randint(2,18,2)
    depths = np.random.randint(2,4,1)
    setpos(loc[0],loc[1], 0 ,s_printer)
    setpos(loc[0], loc[1], -depths[0], s_printer)
    time.sleep(1)
    b = read_sensor(s_sensor)
    b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
    b15 = b15 / norm_val
    #print(b15)
    single_set = [torch.tensor(b15, dtype=torch.float32), torch.tensor(truths[0], dtype=torch.float32)]
    xyF = model(single_set[0])
    truth = [read_printer(s_printer)[0],read_printer(s_printer)[1],read_force(s_Force)]
    print(f"Predicted [X, Y, F] {xyF.detach().numpy()}")
    print(f"Real      [X, Y, F] {truth}, (Z: {read_printer(s_printer)[2]}")
    time.sleep(3)
    setpos(loc[0], loc[1], 0, s_printer)
    setpos(0, 0, 0, s_printer)
    time.sleep(1)

