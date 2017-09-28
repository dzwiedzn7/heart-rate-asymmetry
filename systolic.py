#!/usr/bin/env python
import glob
import csv
import numpy as np
from scipy.signal import argrelextrema
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


def pulse_pressure(Max, Min):
    Max = np.mean(np.array(Max))
    Min = np.mean(np.array(Min))
    return Max - Min


files = glob.glob("*.txt")
"""
cs = csv.writer(open("Descriptors.csv", "a"))
cs.writerow(["nazwa", "SDNN", "SD1", "SD2", "SD1l", "SD1a", "SD1d", "C1a",
             "C1d", "short-condition1", "short-condition2", "SD2a", "SD2d",
             "C2a", "C2d", "Ca", "Cd", "SDNNa", "SDNNd", "long-conditon1",
             "long-condition2", "N", "Nd", "mean","Pulse_Pressure",
             "Double_product"])
"""
# cs = csv.writer(open("Pulse_Pressure.csv", "a"))
# cs.writerow(["nazwa", "Pulse_Pressure"])
cs = csv.writer(open("Double_product.csv", "a"))
cs.writerow(["nazwa","Double_product"])
files2 = [
    open(filename, 'r').readlines()
    for filename in files]

for filename in files:
    data = load_data(from_file=filename)

    data_av = moving_average(data, 15)

    iMax = argrelextrema(data_av, np.greater_equal)[0]
    iMin = argrelextrema(data_av, np.less_equal)[0]

    Max = [data_av[i] for i in iMax]
    Min = [data_av[i] for i in iMin]

    RR = [
        iMax[i]-iMax[i-1]
        for i in np.arange(1, len(iMax))
        if iMax[i]-iMax[i-1] > 70]

    with open('bb_puls_dist.rea', 'a') as plik:
        for i in RR:
            plik.write(str(i)+'\n')
    try:
        pulse = pulse_pressure(Max, Min)
        print pulse, 'pulse_pressure'
        dane = np.array(RR)
        Double_product = np.mean(Max) * (60000.0/np.mean(pulse))
        y = dane[1:]
        x = dane[:-1]
        N = len(dane)
        Nd = deceleration(x, y)
        mean = np.mean(dane)
        SDNN = np.std(dane)
        SD1 = np.std((x-y)/np.sqrt(2))
        SD2 = np.std((x+y)/np.sqrt(2))
        n = len(x)
        SDNN = SDNN*(n-1/n)**(1/2)
        SD1 = SD1*(n-1/n)**(1/2)
        SD2 = SD2*(n-1/n)**(1/2)
        print SD1, "SD1"
        print SD2, "SD2"
        SD1l = (sum((x-y)**2)/2)/n
        SD1l = np.sqrt(SD1l)
        """
        ASYMMETRIC
        """
        """
        short-term
        """
        xy = ((x-np.mean(x)-y+np.mean(y))/np.sqrt(2))
        dec = filter(lambda x: x < 0, xy)
        acc = filter(lambda x: x > 0, xy)
        SD1a = np.sqrt(sum(np.array(acc)**2)/n)
        print SD1a, "SD1a"
        SD1d = np.sqrt(sum(np.array(dec)**2)/n)
        print SD1d, "SD1d"
        short = SD1d**2 + SD1a**2 - SD1l**2
        short2 = SD1d**2 + SD1a**2 - SD1**2
        print round(short, 2), round(short2, 2), "short-term conditions"
        C1a = (SD1a/SD1)**2
        C1d = (SD1d/SD1)**2
        print C1a, "C1a", C1d, "C1d"

        """
        long-term
        """
        XY = (x-np.mean(x)+y-np.mean(y))/np.sqrt(2)
        nochange = [i for i, x in enumerate(xy) if x == 0]
        nacc = [i for i, x in enumerate(xy) if x > 0]
        ndec = [i for i, x in enumerate(xy) if x < 0]
        SD2a = np.sqrt(1.0/n*(sum(XY[nacc]**2)+(sum(XY[nochange]**2)/2.0)))
        SD2d = np.sqrt(1.0/n*(sum(XY[ndec]**2)+(sum(XY[nochange]**2)/2.0)))
        print SD2a, "SD2a"
        print SD2d, "SD2d"
        C2a = (SD2a/SD2)**2
        C2d = (SD2d/SD2)**2
        print C2a, "C2a"
        print C2d, "C2d"
        SDNNa = np.sqrt((SD1a**2+SD2a**2)/2.0)
        SDNNd = np.sqrt((SD1d**2+SD2d**2)/2.0)
        Ca = (SDNNa/SDNN)**2
        Cd = (SDNNd/SDNN)**2
        print Ca, "Ca", Cd, "Cd"
        conditionlong1 = SD2a**2 + SD2d**2 - SD2**2
        conditionlong2 = SDNNa**2+SDNNd**2-SDNN**2
        print round(conditionlong1, 2), round(conditionlong2, 2),
        print "long-term condition"
        # cs.writerow([filename, pulse])
        cs.writerow([filename,Double_product])
    except ZeroDivisionError:
        pass

"""
        cs.writerow([filename, SDNN, SD1, SD2, SD1l, SD1a, SD1d, C1a, C1d,
                     short, short2, SD2a, SD2d, C2a, C2d, Ca, Cd, SDNNa, SDNNd,
                     conditionlong1, conditionlong2, N, Nd, mean, pulse,
                      Double_product])
"""

"""
           plt.plot(data)
           plt.plot(moving_average(data,15))
           plt.scatter(iMax,Max)
           plt.show()
         """

# Bartosz Biczuk
