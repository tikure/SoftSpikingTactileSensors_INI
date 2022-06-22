import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime
#TODO change how files are named
sys.path.append("./Functions")

from SensorCollectionFunctions import *

s_sensor = serial.Serial(port = "COM14", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
for i in range(10):
    print(read_sensor(s_sensor))
    print(len(read_sensor(s_sensor)))
