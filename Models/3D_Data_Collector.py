import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime

sys.path.append("./Functions")

from SensorCollectionFunctions import *

filename = "_real_piezo2"
z_offset = 2
s_sensor = serial.Serial(port="COM5", baudrate=115200, bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_printer = serial.Serial(port="COM4", baudrate=250000)
# s_AFG = serial.Serial(port = "COM10", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
s_piezo = serial.Serial(port="COM3", baudrate=9600)

feedrate = "1600"
initialize_printer(s_printer)
time.sleep(10)
print("Move Printer Check: New Pos = 10/10/0")
setpos(10, 10, 0, s_printer)
b20 = read_sensor(s_sensor)
pos = read_printer(s_printer)
F = read_force(s_piezo)
print("Read Printer Check: ", pos)
print("Read AFG Check: ", F)
print("Read Sensor Check: ", b20)
setpos(0, 0, 0, s_printer)  # Return to origin, not touching pad
if len(b20) != 20 or F != 0 or pos != [10.0, 10.0, 0.0]:
    raise Exception("Sensory Data Incorrect")
setpos(0, 0, 0, s_printer)  # Return to origin, not touching pad

"""Force Sensor Calibration"""
print("Calibrating Force")
setpos(10, 10, -0.5, s_printer)
time.sleep(2)

z_depths = np.arange(0, 1.8+z_offset, 0.05)
forces = []

for z in z_depths:
    setpos(10, 10, -z, s_printer)
    forces.append(read_force(s_piezo))
    # print(read_printer(s_printer))

setpos(10, 10, 0, s_printer)
setpos(0,0,0,s_printer)
plt.vlines(x=z_offset,ymin = 0, ymax = max(forces)+1,colors="r")
plt.plot(z_depths, forces)
plt.title("Force Sensor Calibration")
plt.xlabel("z_depth")
plt.ylabel("Force [N]")
plt.show()
print("----------Initial Checks Completed, commencing data collection------------------\n\n")

"""Normalization Values collected before and after"""
norm_data = []
setpos(0, 0, 0, s_printer)  # Return to origin, not touching pad
time.sleep(1)
print("Collecting Normalization Data")
norm_count = 10000
for i in range(int(norm_count/2)):
    b20 = read_sensor(s_sensor)
    if i % 5000 == 0:
        print(i)
        print("B20 Read")
        print(b20)
    norm_data.append(b20)

"""Avoid corners with screws"""
notest = []
for i in (0, 1, 7, 8):
    for q in (0, 1, 7, 8):
        notest.append([i, q])
notest.append([0, 5, 3])
notest = []

truths = []
sensor_data = []
setpos(0, 0, 0, s_printer)

"""Collect Data"""
iterations = 100
# print(f"Estimated time to completion: {round(170*iterations/60,0)}min")
grid_x = 9  # Steps for sampling + 1 due to indexing
grid_y = 9  # = grid_x, normally
jump_mm = 18 / grid_x
z_depths = [0.4, 0.6, 0.8, 1, 1.2, 1.4]  # how much to indent sensor
z_depths_mm = [-z - z_offset for z in z_depths]  # conversion to mm (z = 0 is 2 mm above sensor)
wait_time = 0.5
setpos(0, 0, 0, s_printer)

time_start = time.time()
time_for_iteration = 0
print(f"Starting Data Collection for {iterations} iterations")
for iteration in range(1, iterations + 1):
    setpos(0, 0, 0, s_printer)
    for g_x in range(1,grid_x+1):  # Iteratate over grid
        g_x_mm = round(g_x * jump_mm, 2)
        setpos(g_x_mm, 0, 0, s_printer)
        for g_y in range(1,grid_y+1):
            g_y_mm = round(g_y * jump_mm, 2)
            if not ([g_x, g_y] in notest):  # Remove corners with screws
                setpos(g_x_mm, g_y_mm, 0, s_printer)  # move above testing depth
                for z_mm in z_depths_mm:
                    setpos(g_x_mm, g_y_mm, z_mm, s_printer)  # move to testing depth
                    # Readdata
                    b20 = read_sensor(s_sensor)
                    sensor_data.append(b20)

                    force_N = read_force(s_piezo)
                    if force_N == 0:  # If not touching sensor we define as 10/10/0 - can be changed later
                        truths.append([10, 10, force_N])
                    else:
                        truths.append([g_x_mm, g_y_mm, force_N])
                        # print([g_x_mm, g_y_mm, force_N])
            else:  # At corners we just take normalization data
                b20 = read_sensor(s_sensor)
                sensor_data.append(b20)
                truths.append([g_x_mm, g_y_mm, 0])
            setpos(g_x_mm, g_y_mm, 0, s_printer)  # Move above sensor again
            #print("x_mm,y_mm,x,y:", g_x_mm, g_y_mm, g_x, g_y)
    if iteration % 20 == 0:
        print(f"Iterations: {iteration}/{iterations}")
        time_for_iteration = round(time.time() - time_start, 1) - time_for_iteration
    if iteration == 1:
        print("Time for Iteration: ", time_for_iteration)
        now = datetime.datetime.now()
        print(now.hour, now.minute)
        print(f"Expected time until completion: {round(time_for_iteration * (iterations - iteration) / 60, 0)}min")
        x = [t[0] for t in truths]
        y = [t[1] for t in truths]
        F = [t[2] for t in truths]
        print(truths)
        print(truths[0])
        print("X:",x)
        print("Y:",y)
        print("F:",F)
        for i in range(len(x)):
            plt.plot(x[i],y[i], "o", markersize= F[i]/2, markerfacecolor='none', markeredgecolor="r")
        plt.title("Data Distribution")
        plt.ylim(0,20)
        plt.xlim(0,20)
        plt.ylabel("Y [mm]")
        plt.xlabel("X [mm]")
        plt.show()
setpos(0, 0, 0, s_printer)

"""Collect Normalization after run"""
# Normalization values:
setpos(0, 0, 0, s_printer)

for i in range(int(norm_count/2)):
    b20 = read_sensor(s_sensor)
    if i % 5000 == 0:
        print(i)
        print("B20 Read")
        print(b20)
    norm_data.append(b20)
print("Total Normalization data: ", len(norm_data))
b15_norm_test = [np.concatenate((b[0:3], b[4:7], b[8:11], b[12:15], b[16:19])) for b in norm_data]
b15_norm_test = np.array(b15_norm_test).T
for val in b15_norm_test:
    plt.plot(val)
plt.title("Normalization Values")
plt.xlabel("Samples")
plt.ylabel("Sensor Data")


"""Save Data"""
print(f"Total time: {round(time.time() - time_start, 0)}s")
print(f"Sensor Data count: {len(sensor_data)}")
if (len(truths) != len(sensor_data)):
    print("---------------------------------------")
    print("---------------------------------------")
    print("Arrays not of equal length")
    print("---------------------------------------")
    print("---------------------------------------")


np.savetxt("./Data/norm_b20_artillery" + filename + ".txt", norm_data, fmt="%s")
np.savetxt("./Data/b20_artillery" + filename + ".txt", sensor_data, fmt="%s")
np.savetxt("./Data/truths_artillery" + filename + ".txt", truths, fmt="%s")
print("Data Saved")
#plt.show()
