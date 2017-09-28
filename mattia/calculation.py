import os
import glob
import csv
import numpy as np
#import runpy
#import app
from Tkinter import *
from tkFileDialog import askdirectory

root = Tk()
root.withdraw()
root.update()
directory = askdirectory()
root.destroy()

files = glob.glob(directory + '/' + '*.csv')
choice = raw_input('Choose a variable [HR,speed,pace,cadence,altitude,temperature,ALL: ]')
ofile  = open(directory + '/' + choice +'results.txt', "wb")
writer = csv.writer(ofile, delimiter=',')#, quotechar='"', quoting=csv.QUOTE_ALL)
if choice == 'HR':
    writer.writerow(['filename','SD','mean','CV'])
    choice = 2
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = float(row.split(',')[choice])
            data.append(row)
        SD = np.std(data)
        mean = np.mean(data)
        CV = SD/mean * 100.0
        writer.writerow([os.path.basename(plik.name),SD,mean,CV])
elif choice == 'speed':
    writer.writerow(['filename','SD','mean','CV'])
    choice = 3
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = float(row.split(',')[choice])
            data.append(row)
        SD = np.std(data)
        mean = np.mean(data)
        CV = SD/mean * 100.0
        writer.writerow([os.path.basename(plik.name),SD,mean,CV])
elif choice == 'pace':
    writer.writerow(['filename','SD','mean','CV'])
    choice = 4
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = row.split(',')[choice]
            (m, s) = row.split(':')
            seconds = int(m) * 60 + int(s)
            row = float(seconds) / 60
            data.append(row)
        SD = np.std(data)
        mean = np.mean(data)
        CV = SD/mean * 100.0
        writer.writerow([os.path.basename(plik.name),SD,mean,CV])
elif choice == 'cadence':
    writer.writerow(['filename','SD','mean','CV'])
    choice = 5
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = float(row.split(',')[choice])
            data.append(row)
        SD = np.std(data)
        mean = np.mean(data)
        CV = SD/mean * 100.0
        writer.writerow([os.path.basename(plik.name),SD,mean,CV])
elif choice == 'altitude':
    writer.writerow(['filename','delta altitude'])
    choice = 6
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = float(row.split(',')[choice])
            data.append(row)
        height = data[-1] - data[0]
        writer.writerow([os.path.basename(plik.name),height])

elif choice == 'temperature':
    writer.writerow(['filename','SD','mean','CV'])
    choice = 9
    for plik in files:
        filename = plik
        data = []
        plik = open(plik,'r')
        for row in plik:
            row = float(row.split(',')[choice])
            data.append(row)
        SD = np.std(data)
        mean = np.mean(data)
        CV = SD/mean * 100.0
        writer.writerow([os.path.basename(plik.name),SD,mean,CV])
elif choice == 'ALL':
        writer.writerow(['filename','HR_mean','HR_std','HR_cv','Speed_mean','Speed_std','Speed_cv','Pace_mean','Pace_std','Pace_cv'])
        for plik in files:
            filename = plik
            plik = open(plik,'r')
            HR,speed,Pace = [],[],[]
            for row in plik:
                HR.append(float(row.split(',')[2]))
                speed.append(float(row.split(',')[3]))
                pac = row.split(',')[4]
                (m, s) = pac.split(':')
                seconds = int(m) * 60 + int(s)
                pac = float(seconds) / 60
                Pace.append(pac)
            HR_mean = np.mean(HR)
            HR_std = np.std(HR)
            HR_cv = (np.std(HR)/np.mean(HR)) * 100.0
            Speed_mean = np.mean(speed)
            Speed_std = np.std(speed)
            Speed_cv = np.std(speed)/np.mean(speed) * 100.0
            Pace_mean = np.mean(Pace)
            Pace_std = np.std(Pace)
            Pace_cv = np.std(Pace)/np.mean(Pace) * 100.0
            writer.writerow([os.path.basename(plik.name),HR_mean,HR_std,HR_cv,
                             Speed_mean,Speed_std,Speed_cv,Pace_mean,Pace_std,Pace_cv])
