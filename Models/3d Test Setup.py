import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime

sys.path.append("./Functions")

from SensorCollectionFunctions import *

#s_sensor = serial.Serial(port = "COM8", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_printer = serial.Serial(port = "COM4", baudrate=250000) #COM4 on MSI
#s_AFG = serial.Serial(port = "COM10", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_piezo = serial.Serial(port = "COM3", baudrate=9600) #COm3 on MSI

feedrate = "1600"
while(1):
    print(read_force(s_piezo))
    time.sleep(0.1)