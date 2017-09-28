import glob
import datetime
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal
from Tkinter import *
from tkFileDialog import askdirectory,askopenfilename
import app

def moving_average(data, window):
    return np.convolve(data, np.ones(window)/window, mode='same')

def filtr(data):
    b, a = signal.butter(2, 0.05)
    return signal.filtfilt(b, a, data)



def check_plot(t,HR,name):
    xs = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')

    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)
    ax.xaxis.set_major_formatter(hfmt)
    plt.setp(ax.get_xticklabels(), rotation=15)
    ax.plot(t, HR)
    plt.show()

def plot_all(t,s,hr,pace,name):
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    pace = matplotlib.dates.date2num(pace)
    lala = matplotlib.dates.DateFormatter('%M:%S')
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    ax.xaxis.set_major_formatter(lala)
    plt.xlabel('Time[min]')
    plt.ylabel('Speed[km/h]')
    ax.plot_date(t,hr,'b-',label = 'HR',color='green')
    plt.plot_date(t, s,fmt='b-', tz=None, xdate=True,
              ydate=False, label="Speed", color='red')
    plt.plot_date(t, pace,fmt='b-', tz=None, xdate=True,
                  ydate=True, label="pace", color='blue')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    #ax.set_ylim([limits[1][1],limits[1][0]])
    plt.gcf().autofmt_xdate()
    fig.savefig(name + '_all_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def plot_speed(t,s,name):
    limits = get_limits()
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    #,fmt='')
    filt = filtr(s)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    plt.xlabel('Time[min]')
    plt.ylabel('Speed[km/h]')
    ax.plot_date(t,filt,'b-',label = 'Filtered Speed')
    plt.plot_date(t, s,fmt='b-', tz=None, xdate=True,
              ydate=False, label="Speed", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    #ax.set_ylim([limits[0],limits[1]])
    plt.gcf().autofmt_xdate()
    fig.savefig(name + '_Speed_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()




def plot_HR(t,HR,name):
    limits = get_limits(get_data)
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    #,fmt='')
    filt = filtr(HR)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    plt.xlabel('Time[min]')
    plt.ylabel('HR[bpm]')
    ax.plot_date(t,filt,'b-',label = 'Filtered HR')
    plt.plot_date(t, HR,fmt='b-', tz=None, xdate=True,
                  ydate=False, label="HR", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    ax.set_ylim([limits[1],limits[0]])
    plt.gcf().autofmt_xdate()
    fig.savefig(name + '_HR_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()
    #plt.show()


def plot_pace(t,pace,name):
    t = matplotlib.dates.date2num(t)
    pace = matplotlib.dates.date2num(pace)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    lala = matplotlib.dates.DateFormatter('%M:%S')

    #,fmt='')
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    ax.xaxis.set_major_formatter(lala)

    plt.xlabel('Time[min]')
    plt.ylabel('Pace')
    plt.plot_date(t, pace,fmt='b-', tz=None, xdate=True,
                  ydate=True, label="pace", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    fig.savefig(name + '_Pace_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

def plot_cadence(t,cadence,name):
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    #,fmt='')
    filt = filtr(cadence)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    plt.xlabel('Time[min]')
    plt.ylabel('Cadence')
    ax.plot_date(t,filt,'b-',label = 'Filtered Cadence')
    plt.plot_date(t, cadence,fmt='b-', tz=None, xdate=True,
                  ydate=False, label="Cadence", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    fig.savefig(name + '_Cadence_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

def plot_altitude(t,altitude,name):
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    #,fmt='')
    #filt = filtr(HR)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    plt.xlabel('Time[min]')
    plt.ylabel('Altitude[m]')
    #ax.plot_date(t,filt,'b-',label = 'Filtered HR')
    plt.plot_date(t, altitude,fmt='b-', tz=None, xdate=True,
                  ydate=False, label="Altitude", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    fig.savefig(name + '_Altitude_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()

def plot_temperature(t,temperature,name):
    t = matplotlib.dates.date2num(t)
    hfmt = matplotlib.dates.DateFormatter('%H:%M:%S')
    #,fmt='')
    #filt = filtr(HR)
    fig = plt.figure(1)
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(hfmt)
    plt.xlabel('Time[min]')
    plt.ylabel('Temperature[C]')
    #ax.plot_date(t,filt,'b-',label = 'Filtered HR')
    plt.plot_date(t, Cadence,fmt='b-', tz=None, xdate=True,
                  ydate=False, label="Cadence", color='red')
    handles, labels = ax.get_legend_handles_labels()
    lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1,0.5))
    fig.savefig(name + '_Cadence_plot' + '.png', bbox_extra_artists=(lgd,), bbox_inches='tight')
    plt.clf()


def get_limits(get_data):
    hr = np.loadtxt(get_data,skiprows=4,delimiter=',',usecols = (2,))
    speed = np.loadtxt(get_data,skiprows=4,delimiter=',',usecols = (3,))
    print hr,speed
    max_hr,min_hr = max(hr),min(hr)
    max_speed,min_speed = max(speed),min(speed)
    return np.array([max_hr,min_hr,max_speed,min_speed])

print askdirectory
root = Tk()
root.withdraw()
root.update()
directory = askdirectory()
root.destroy()


root = Tk()
root.withdraw()
root.update()
get_data = askopenfilename()
root.quit()
files = glob.glob(directory + '/' + '*.csv')
for plik in files:
    filename = plik
    lim = []
    time,HR,speed,pace,cadence,altitude,temperature = [],[],[],[],[],[],[]
    plik = open(plik,'r').readlines()
    for line in plik:
        line = line.split(',')
        #tt = line[1]
        #(h, m, s) = tt.split(':')
        #seconds = int(h) * 3600 +int(m) * 60 + int(s)
        #row = float(seconds) / 60
        #pac = line[4]
        #(m,s) = pac.split(':')
        #secs = int(m) * 60 + int(s)
        #secs = float(secs)/60
        time.append(line[1])
        HR.append(float(line[2]))
        speed.append(float(line[3]))
        pace.append(line[4])
        cadence.append(float(line[5]))
        altitude.append(float(line[6]))
        temperature.append(float(line[9]))
    time = [datetime.datetime.strptime(i,'%H:%M:%S' ) for i in time]
    pace = [datetime.datetime.strptime(i,'%M:%S' ) for i in pace]
    collection = np.array([time,HR,speed,pace,cadence,altitude,temperature])
    if app.z == 'HR/time':
        plot_HR(collection[0],collection[1],filename)
    if app.z == 'Speed/time':
        plot_speed(collection[0],collection[2],filename)
    if app.z == 'Pace/time':
        plot_pace(collection[0],collection[3],filename)
    if app.z == 'Cadence/time':
        plot_cadence(collection[0],collection[4],filename)
    if app.z == 'Altitude/time':
        plot_altitude(collection[0],collection[5],filename)
    if app.z == 'Temperature/time':
        plot_temperature(collection[0],collection[6],filename)
    if app.z =='All':
        plot_all(collection[0],collection[1],collection[2],collection[3],filename)
