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

b20_norm = 
