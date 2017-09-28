import sys,os
import glob
import csv
import numpy as np
import matplotlib.pyplot as plt
from pyexcel.cookbook import merge_all_to_a_book
import pyexcel.ext.xlsx


files = glob.glob('*.csv')
choice1=raw_input('How do you want to cut the files [time/dist]? :')
if choice1 == 'time':
    choice2 = int(raw_input('Choose the length of the pieces (1,2,3,4,5 [minutes]): '))
    choice2 = choice2 * 60
    for plik in files:
        os.makedirs('#time#' + plik.rstrip('.csv'))
        runner = open(plik,'r').readlines()[4:]
        filename = 1
        for i in range(len(runner)):
            if i % choice2 == 0:
                open(os.getcwd() + '/' +'#time#' + plik.rstrip('.csv') +'/' +  str(filename) \
                + '#time#' + plik, 'w+').writelines(runner[i:i+choice2])
                filename += 1
        merge_all_to_a_book(glob.glob('#time#' + plik.rstrip('.csv') + '/' + "*.csv"),str(choice2) + plik.rstrip('.csv') + "output.xlsx")
elif choice1 == 'dist':
    choice2 = int(raw_input('Choose the length of the pieces (500,1000,2000,5000 [metres]): '))
    for plik in files:
        os.makedirs('#dist#' + plik.rstrip('.csv'))
        runner = open(plik,'r').readlines()[4:]
        filename = 1
        counter = []
        for idx,val in enumerate(runner):
            count = float(val.split(',')[8])
            counter.append(count)
            if round(count,-1) % choice2 == 0:
                if len(counter) > 10:
                    open(os.getcwd() + '/' + '#dist#' + plik.rstrip('.csv') +'/'  +str(choice2 *filename) \
                    + '#dist#' + plik, 'w+').writelines(runner[idx - len(counter)-1:idx+1])
                    var = open(os.getcwd() + '/' +'#dist#' + plik.rstrip('.csv') +'/' +  str(choice2 *filename) \
                          + '#dist#' + plik,'r')
                    var_path = os.path.realpath(var.name)
#
                    if os.path.getsize(var_path) < 100:
                        os.remove(var_path)
                    filename += 1

                else:
                    open(os.getcwd() + '/' +'#dist#' + plik.rstrip('.csv') +'/' +str(choice2 *filename) \
                          + '#dist#' + plik,'a+').writelines(runner[idx - len(counter)-1:idx+1])
                    var = open(os.getcwd() + '/' +'#dist#' + plik.rstrip('.csv') +'/' +  str(choice2 *filename) \
                          + '#dist#' + plik,'r')
                    var_path = os.path.realpath(var.name)
#
                    if os.path.getsize(var_path) < 100:
                        os.remove(var_path)
                del counter[:]
            elif idx == len(runner) -1:
                open(os.getcwd() + '/' +'#dist#' + plik.rstrip('.csv') +'/' +str(choice2 *filename) \
                      + '#dist#' + plik,'a+').writelines(runner[idx - len(counter)-1:idx+1])
        merge_all_to_a_book(glob.glob('#dist#' + plik.rstrip('.csv') + '/' + "*.csv"),str(choice2) + plik.rstrip('.csv') + "output.xlsx")
