import serial
import matplotlib.pyplot as plt
import time
import numpy as np

"""Mecmesin Advanced Force Gauge Code"""

#s_AFG.write(0x2A)#* 42 0x2A Continuous transmit
def read_force(s_AFG):
    """Reads the force from Mecmesin AFG on s_AFG"""
    #time.sleep(0.05)
    s_AFG.flushInput()
    fin = True
    while(fin):
        s_AFG.flushInput()
        force_N = s_AFG.readline().decode().strip()
        force_N2 = s_AFG.readline().decode().strip()
        while force_N == '':  # Incase empty bit arrives
            s_AFG.flushInput()
            #s_AFG.write(0x3F)  # ? 63 0x3F Transmit the current reading
            force_N = s_AFG.readline().decode().strip()
        force_N = round(abs(float(force_N)),3)
        time.sleep(0.01)
        while force_N2 == '':  # Incase empty bit arrives
            s_AFG.flushInput()
            #s_AFG.write(0x3F)  # ? 63 0x3F Transmit the current reading
            force_N2 = s_AFG.readline().decode().strip()
        force_N2 = round(abs(float(force_N2)),3)
        if abs(force_N - force_N2) <= force_N*0.1:
            fin = False
            #print(f"Forces accepted: {force_N}, {force_N2}, {abs(force_N - force_N2)}, {force_N * 0.1}")
        else:
            print(f"Forces rejected: {force_N}, {force_N2}, {abs(force_N - force_N2)}, {force_N * 0.1}")
    return force_N

def read_force_OFF(s_piezo):
    """Reads Force from piezo sensor via arduino serial"""
    s_piezo.flushInput()
    force_IDK = s_piezo.readline().decode().strip()
    while force_IDK == "":
        force_IDK = s_piezo.readline().decode().strip()
    return int(force_IDK)


def read_printer(s_printer):
    """Reads printers current position over serial"""
    s_printer.flushInput()
    s_printer.write("M114 R\n".encode())#M114 returns cooridnated
    grbl_out = s_printer.readline().decode()
    #print(grbl_out)
    while(grbl_out[0] != "X"):#Sometimes temperature readings get transmitted instead
        grbl_out = s_printer.readline().decode()
    #print(grbl_out)
    data = grbl_out.split()[0:3]#X, Y, Z
    data = [float(d[2:]) for d in data]
    return data


def read_sensor(s_sensor):
    """Reads ReSkin Sensor from arduino over serial"""
    s_sensor.flushInput()
    serialString = s_sensor.readline()
    serialString = serialString.decode('Ascii')
    # time.sleep(0.001)

    b20 = [float(b) for b in serialString.split()]
    return b20


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
    print(read_printer(s_printer))
    time.sleep(5)
    #setpos(0,0,0,s_printer)
    print('Sending: ' + "G92")
    s_printer.write("G92 X0 Y0 Z0\n".encode())
    print(read_printer(s_printer))
    time.sleep(1)