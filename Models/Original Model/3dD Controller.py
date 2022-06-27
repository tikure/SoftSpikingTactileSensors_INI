import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime
import torch
import torch.nn as nn

sys.path.append("./Functions")

def setpos(X,Y,Z,s_printer):
    feedrate = "1600"
    """Sets position of 3d printer via serial (Marlin)"""
    line = "G0 "+ " X"+str(X)+" Y"+str(Y)+ " Z"+str(Z)+  " F" + str(feedrate)+"\n"
    s_printer.write(line.encode()) # Send g-code block to grbl
    line = "M400 \n"
    s_printer.write(line.encode())#M400 halts gcode until move is completed
    line = "M118 moved!\n"
    s_printer.write(line.encode()) #M118 asks serial to send from 3d (but only after M400/move is completed)
    grbl = s_printer.readline().decode()
    while grbl != 'moved!\r\n':#Dont continue if we have not moved
        grbl = s_printer.readline().decode()
    return

def initialize_printer(s_printer):
    """Sets origin for printer - set position to be above bottom left screw and 2mm above sensor surface (X & Y + 10)"""
    # Setup
    # Wake up grbl
    s_printer.write("\r\n\r\n".encode())
    time.sleep(2)  # Wait for grbl to initialize
    s_printer.flushInput()  # Flush startup text in serial input

    # Set established 0/0/0 pos
    feedrate = "1600"

    print('Sending: ' + "G90")
    s_printer.write("G90\n".encode())

    # setpos(x_def, y_def, z_def)
    #print(read_printer(s_printer))
    time.sleep(5)
    #setpos(0,0,0,s_printer)
    print('Sending: ' + "G92")
    s_printer.write("G92 X0 Y0 Z0\n".encode())
    #print(read_printer(s_printer))
    time.sleep(1)

s_printer = serial.Serial(port="COM8", baudrate=250000)
#s_Force = serial.Serial(port = "COM7", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
#s_sensor = serial.Serial(port="COM5", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)

feedrate = "1600"
setpos(0,0,0, s_printer)
initialize_printer(s_printer)
time.sleep(1)

truths = [0,0,0]
while 1:
    print("---------------")
    loc = np.random.randint(2,18,2)
    depths = np.random.randint(3,4,1)
    setpos(loc[0],loc[1], 0 ,s_printer)
    setpos(loc[0], loc[1], -depths[0], s_printer)
    setpos(loc[0], loc[1], 0, s_printer)
    setpos(0, 0, 0, s_printer)
    time.sleep(1)