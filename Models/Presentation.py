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
from Data_functions import *

"""Import Training Data"""""
#filenames = ["_AFG_Board2_50","_AFG_Board2_50_2"]
filenames = ["_AFG_board2_50_screw","_AFG_board2_50_2_screw"]
model_name = "_AFG_board2_screw_2"
b15, truths, test_truths, norm_val, b15_norm = import_data(filenames, max_N=100, shape="random", include_norm = False,
                                                 normalization ='divisive', data_count_percent = 100)


b15_norm = b15_norm[0:1000]

s_sensor = serial.Serial(port="COM3", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
for i in range(1000):
    b = read_sensor(s_sensor)
    b15  = np.array(np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])))
    b15_norm.append(b15)
    if i%10 == 0:
        print("done")




x = [[b[0] for b in b15_norm], [b[3] for b in b15_norm], [b[6] for b in b15_norm], [b[9] for b in b15_norm],
     [b[12] for b in b15_norm]]
y = [[b[1] for b in b15_norm], [b[4] for b in b15_norm], [b[7] for b in b15_norm], [b[10] for b in b15_norm],
     [b[13] for b in b15_norm]]
z = [[b[2] for b in b15_norm], [b[5] for b in b15_norm], [b[8] for b in b15_norm], [b[11] for b in b15_norm],
     [b[14] for b in b15_norm]]

fig, axs = plt.subplots(3, 1, sharex=True)
fig.suptitle('Normalization Data')

axs[0].set_ylabel("X Sensor data")
axs[1].set_ylabel("Y Sensor data")
axs[2].set_ylabel("Z Sensor data")
axs[2].set_xlabel("Samples")
for x_n in x:
    axs[0].plot(x_n, ",")
for y_n in y:
    axs[1].plot(y_n, ",")
for i, z_n in enumerate(z):
    axs[2].plot(z_n, ",", label="S" + str(i + 1))
plt.legend(loc='upper right', prop={'size': 5})
plt.show()

