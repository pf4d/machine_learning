# genetic_algorithm.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the genetic algorithm to solve the 'knapsack' maximization
# problem.

import csv
import sys

import matplotlib.gridspec as gridspec

from time       import time
from pylab      import *
from scipy.misc import factorial as fact

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
# collect the data :
# columns : Weight, Price

dirc  = '../data/'       # directory containing data

attrib  = {0 : 'Weight', 
           1 : 'Price'}

# dataset :
f    = open(dirc + 'items.csv', 'r')
data = loadtxt(f, dtype='float', delimiter=",", skiprows=1)

maxW = 200.0 

#===============================================================================
# perform classification with k-fold cross-validation :



