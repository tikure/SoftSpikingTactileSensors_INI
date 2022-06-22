import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime
#TODO change how files are named
sys.path.append("./Functions")

from SensorCollectionFunctions import *
from Model_functions import *

s_Force = serial.Serial(port = "COM7", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_sensor = serial.Serial(port="COM5", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_printer = serial.Serial(port="COM4", baudrate=250000)

setpos(10,10,-3, s_printer)
print(read_force(s_Force))