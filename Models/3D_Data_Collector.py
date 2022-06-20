import serial
import matplotlib.pyplot as plt
import time
import numpy as np
import sys
import datetime

sys.path.append("./Functions")

from SensorCollectionFunctions import *

s_sensor = serial.Serial(port = "COM8", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
#s_printer = serial.Serial(port = "COM11", baudrate=250000)
#s_AFG = serial.Serial(port = "COM10", baudrate=115200,bytesize=8, timeout=2, stopbits=serial.STOPBITS_ONE)
#s_piezo = serial.Serial(port = "COM6", baudrate=9600)

feedrate = "1600"
initialize_printer(s_printer)

print("Move Printer Check: New Pos = 10/10/0")
setpos(10,10,0,s_printer)
print("Read Printer Check: ", read_printer(s_printer))
print("Read AFG Check: ", read_force(s_piezo))
print("Read Sensor Check: ", read_sensor(s_sensor))
print("----------Initial Checks Completed, commencing data collection------------------\n\n")


"""Normalization Values collected before and after"""
norm_data = []
setpos(0,0,0)#Return to origin, not touching pad
time.sleep(1)

for i in range(5000):
    b20 = read_sensor(s_sensor)
    if i%5000== 0:
        print(i)
        print("B20 Read")
        print(b20)
    norm_data.append(b20)


"""Avoid corners with screws"""
notest = []
for i in (0,1,7,8):
    for q in (0,1,7,8):
        notest.append([i,q])
notest.append([0,5,3])
print(len(notest))
print(notest)
print([0,5] in notest)

truths = []
sensor_data = []
setpos(0,0,0,s_printer)

"""Collect Data"""
iterations = 200
# print(f"Estimated time to completion: {round(170*iterations/60,0)}min")
grid_x = 9 #Steps for sampling
grid_y = 9 # = grid_x, normally
jump_mm = 19 / grix_x
z_depths = [0.4, 0.6, 0.8, 1, 1.2, 1.4] #how much to indent sensor
z_depths_mm = [-z - 2.4 for z in z_depths] #conversion to mm (z = 0 is 2 mm above sensor)
wait_time = 0.5
setpos(0, 0, 0, s_printer)

time_start = time.time()
time_for_iteration = 0
for iteration in range(1, iterations):
    setpos(0, 0, 0, s_printer)
    for g_x in range(grid_x):  # Iteratate over grid
        g_x_mm = round(g_x * jump_mm, 2)
        setpos(g_x_mm, 0, 0, s_printer)
        for g_y in range(grid_y):
            if not ([g_x, g_y] in notest):  # Remove corners with screws
                g_y_mm = round(g_y * jump_mm, 2)
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
                        #print([g_x_mm, g_y_mm, force_N])
            else: #At corners we just take normalization data
                b20 = read_sensor()
                sensor_data.append(b20)
                truths.append([-1, -1, 0])
            setpos(g_x_mm, g_y_mm, 0)  # Move above sensor again

    if iteration%20 = 0:
    print(f"Iterations: {iteration}/{iterations}")

    time_for_iteration = round(time.time() - time_start, 0) - time_for_iteration
    if iteration == 1:
        print("Time for Iteration: ", time_for_iteration)
        now = datetime.datetime.now()
        print(now.hour, now.minute)
        print(f"Expected time until completion: {round(time_for_iteration * (iterations - iteration) / 60, 0)}min")
setpos(0, 0, 0)

"""Collect Normalization after run"""
#Normalization values:
setpos(0,0,0,s_printer)

for i in range(5000):
    b20 = read_sensor()
    if i%5000 == 0:
        print(i)
        print("B20 Read")
        print(b20)
    norm_data.append(b20)
print("Total Normalization data: ",len(norm_data))
b15_norm_test = [np.concatenate((b[0:3],b[4:7],b[8:11],b[12:15],b[16:19])) for b in norm_data]
b15_norm_test = np.array(b15_norm_test).T
for val in b15_norm_test:
    plt.plot(val)
plt.show()

"""Save Data"""
print(f"Total time: {round(time.time()-time_start,0)}s")
print(f"Sensor Data count: {len(sensor_data)}")
if(len(truths) != len(sensor_data)):
    print("---------------------------------------")
    print("---------------------------------------")
    print("Arrays not of equal length")
    print("---------------------------------------")
    print("---------------------------------------")

filename = "_piezo"
np.savetxt("./Data/norm_b20_artillery"+filename+".txt", norm_data, fmt="%s")
np.savetxt("./Data/b20_artillery"+filename+".txt",, sensor_data, fmt="%s")
np.savetxt("./Data/truths_artillery"+filename+".txt", truths, fmt="%s")


