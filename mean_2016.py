#!/usr/bin/env python
import glob
import csv
import numpy as np
from scipy.signal import argrelextrema
from scipy.integrate import simps
import pylab as plt


PULSE_WAVE_PRESSURE_COLUMN = 1
PREFIX_LINES = 4


def load_data(from_file):

    with open(from_file, 'r') as f:
        lines = f.readlines()

    data_lines = [
        line.replace(',', '.').split()
        for line in lines[PREFIX_LINES:]]

    pulse_wave_pressure = [
        float(columns[PULSE_WAVE_PRESSURE_COLUMN])
        for columns in data_lines
        if len(columns) > PULSE_WAVE_PRESSURE_COLUMN]

    return np.array(pulse_wave_pressure)


def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='valid')


def deceleration(x, y):
    dif = np.array(x) - np.array(y)
    ND = [i for i in dif if i > 0]
    return len(ND)


def integral(samples):
    return simps(samples)

def simpson(data):
    data = np.array(data)
    a = 0
    b = len(data)
    n = len(data)
    #if n%2 == 1:
        #n = n-1
    h = (b-a)/n
    result = 0
    for i in range(1,n, 2):
        result += 4*data[i]*h
    for i in range(2,n-1, 2):
        result += 2*data[i]*h
    return result * h /3




"""
cs = csv.writer(open("Cardiac_Cycle_asymmetric.csv", "a"))
cs.writerow(["name","Mean_blood_pressure","Diastolic","Systolic",
             "SDNN", "SD1", "SD2", "SD1l", "SD1a", "SD1d", "C1a",
             "C1d", "short-condition1", "short-condition2", "SD2a", "SD2d",
             "C2a", "C2d", "Ca", "Cd", "SDNNa", "SDNNd", "long-conditon1",
             "long-condition2", "N", "Nd", "mean", "mean_cardiac_cycle",
             "average_pulse_interval"])

cs2 = csv.writer(open("Mean_blood_pressure.csv", "a"))
cs2.writerow(["name", "Mean_blood_pressure",
             "Systolic", "Diastolic"])
"""
cs3 = csv.writer(open("Pulse_Pressure.csv", "a"))
cs3.writerow(["name", "SDNN", "SD1", "SD2", "SD1l", "SD1a", "SD1d", "C1a",
              "C1d", "C1d+C1a", "SD2a", "SD2d",
              "C2a", "C2d", "Ca", "Cd", "SDNNa", "SDNNd", "C2a+C2d",
              "Ca+Cd", "N", "mean"])


files = glob.glob("*.txt")
files2 = [
    open(filename, 'r').readlines()
    for filename in files]

for filename in files:
    data = load_data(from_file=filename)

    data_av = moving_average(data, 15)

    iMax = argrelextrema(data_av, np.greater_equal)[0]
    iMax = [iMax[i] for i in range(1,len(iMax)) if iMax[i] - iMax[i-1] > 2]

    iMin = argrelextrema(data_av, np.less_equal)[0]
    iMin = [iMin[i] for i in range(1,len(iMin)) if iMin[i] - iMin[i-1] > 2]

    Max = [data_av[i] for i in iMax if data_av[i] > 50]
    iMax = [i for i in iMax if data_av[i] >50]

    Min = [data_av[i] for i in iMin]

    IBI = [iMax[i]-iMax[i-1] for i in np.arange(1, len(iMax))
           if iMax[i]-iMax[i-1] > 20]

    HR = [60000/i for i in IBI]

    RPP = [i*j for i, j in zip(HR, Max)]

    Pulse_Pressure = [i-j for i, j in zip(Max, Min)]
    Pulse_Pressure = [i for i in Pulse_Pressure if i > 20]

    Cardiac_Cycle = np.array([
                    iMin[i] - iMin[i-1]
                    for i in range(1, len(iMin))
                    if iMin[i] - iMin[i-1] > 20])
    try:
        integral_area = np.array(filter(lambda x: len(x) > 5,
                                 [data_av[iMin[i]:iMin[i+1]+1]
                                 for i in range(len(iMin)-1)]))

        mean_cardiac = np.mean(Cardiac_Cycle)
        pulse_interval = np.array([i * 1.0/120.0 for i in Cardiac_Cycle])
        mean_pulse_interval = np.mean(pulse_interval)
        mean_blood_pressure_V2 = np.array([integral(i)*(1.0/len(i))

                                       for i in integral_area])
        mean_blood_pressure = np.array([simpson(i)*(1.0/len(i))
                                          for i in integral_area])
        #cs2.writerow([filename])
        #for i in range(len(Cardiac_Cycle)):
        #    cs2.writerow(["", mean_blood_pressure[i], Max[i], Min[i]])
        #plt.plot(range(len(data)), data)
        plt.plot(range(len(data_av)), data_av)
        #plt.scatter(iMax,Max,color='magenta')
        #plt.scatter(iMin,Min,color='red')
        #plt.title(filename)
        #plt.show()
        dane = np.array(Pulse_Pressure)
        y = dane[1:]
        x = dane[:-1]
        N = len(dane)
#        Nd = deceleration(x, y)
        mean = np.mean(dane)
        SDNN = np.std(dane)
        SD1 = np.std((x-y)/np.sqrt(2))
        SD2 = np.std((x+y)/np.sqrt(2))
        n = len(x)
        SDNN = SDNN*(n-1/n)**(1/2)
        SD1 = SD1*(n-1/n)**(1/2)
        SD2 = SD2*(n-1/n)**(1/2)
        SD1l = (sum((x-y)**2)/2)/n
        SD1l = np.sqrt(SD1l)

        # ASYMMETRIC

        # short-term
        xy = ((x-np.mean(x)-y+np.mean(y))/np.sqrt(2))
        dec = filter(lambda x: x < 0, xy)
        acc = filter(lambda x: x > 0, xy)
        SD1a = np.sqrt(sum(np.array(acc)**2)/n)
        SD1d = np.sqrt(sum(np.array(dec)**2)/n)
        #short = SD1d**2 + SD1a**2 - SD1l**2
        #short2 = SD1d**2 + SD1a**2 - SD1**2
        C1a = (SD1a/SD1)**2
        C1d = (SD1d/SD1)**2
        Cshort = C1a + C1d
        # long-term
        XY = (x-np.mean(x)+y-np.mean(y))/np.sqrt(2)
        nochange = [i for i, x in enumerate(xy) if x == 0]
        nacc = [i for i, x in enumerate(xy) if x > 0]
        ndec = [i for i, x in enumerate(xy) if x < 0]
        SD2a = np.sqrt(1.0/n*(sum(XY[nacc]**2)+(sum(XY[nochange]**2)/2.0)))
        SD2d = np.sqrt(1.0/n*(sum(XY[ndec]**2)+(sum(XY[nochange]**2)/2.0)))
        C2a = (SD2a/SD2)**2
        C2d = (SD2d/SD2)**2
        SDNNa = np.sqrt((SD1a**2+SD2a**2)/2.0)
        SDNNd = np.sqrt((SD1d**2+SD2d**2)/2.0)
        SDNN = SDNNa**2 + SDNNd**2
        Ca = SDNNa**2/SDNN
        Cd = SDNNd**2/SDNN
        #conditionlong1 = SD2a**2 + SD2d**2 - SD2**2
        #conditionlong2 = SDNNa**2+SDNNd**2-SDNN**2
        Clong = C2a + C2d
        Ctotal = Ca + Cd
        cs3.writerow([filename, SDNN, SD1, SD2, SD1l, SD1a, SD1d, C1a, C1d,
                    Cshort, SD2a, SD2d, C2a, C2d, Ca, Cd, SDNNa, SDNNd,
                    Clong,Ctotal, N, mean])
        #cs2.writerow([filename])

    except ZeroDivisionError,IndexError:
        pass
