# cand_elim.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the candidate elimination method to classify a set of data.  
# The accuracy is reported along with the time to compute.
#

import csv
import sys

import matplotlib.gridspec as gridspec

from time  import time
from pylab import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
# collect the data :
# columns : Sky, AirTemp, Humidity, Wind, Water, Forecast, class

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {0 : 'Do Not Enjoy', 1 : 'Enjoy Sport'}
attrib  = {0 : 'Sky', 
           1 : 'Air Temp',
           2 : 'Humidity', 
           3 : 'Wind', 
           4 : 'Water', 
           5 : 'Forecast'}
iattrib = {'Sky'      : 0, 
           'Air Temp' : 1,
           'Humidity' : 2, 
           'Wind'     : 3, 
           'Water'    : 4, 
           'Forecast' : 5}


# dataset :
f    = open(dirc + 'trainingDataCandElim.csv', 'r')
data = loadtxt(f, dtype='str', delimiter=",", skiprows=1)


#===============================================================================
# functions used to solve :
def plot_results(test_class, v, name, label=None):
  """
  plot the absolute value of the differece between the true class <test_class>
  and classified class <v> with classifier acronym <name>.
  """
  fig = figure(figsize=(12,4))
  plot(abs(test_class - v), 'k', drawstyle='steps-mid', lw=2.0, label=label)
  for x,c in zip(range(len(test_class)), test_class):
    if c == 4: 
      text(x, 0.0, c, horizontalalignment='center',
                      verticalalignment='center')
  tight_layout()
  title(r'$|v - v_{%s}|$' % name)
  xlabel(r'$n$')
  ylim([-1, 2])
  grid()
  if label != None: legend()
  savefig('../doc/images/' + name + '_results.png', dpi=300)
  show()


#===============================================================================
# perform classification :

# get the unique values of each column:
def find_unique_col(S):
  """
  find the unique values of each column of array <S>.
  """
  unq = []
  for i in range(shape(S)[1]):
    unq.append(unique(S[:,i]))
  return array(unq)

def find_unique_row(S):
  """
  find the unique values of each row of array <S>.
  """
  unq = unique(S.view(S.dtype.descr * S.shape[1]))
  return unq.view(S.dtype).reshape(-1, S.shape[1])

def get_wild(unq, unq_i):
  """
  determine if unique set <unq> is the same as unique set <unq_i>.
  """
  wild = []
  for uc, up in zip(unq, unq_i):
    if len(uc) == len(up): wild.append(True)
    else:                  wild.append(False)
  return array(wild)

def get_spec(unq_i, unq_j):
  """ 
  determine if any element of <unq_i> is in <unq_j>.  If so, element is False,
  otherwise element is True.
  """
  spec = []
  for ui, uj in zip(unq_i, unq_j):
    if len(intersect1d(ui, uj)) != 0:
      spec.append(False)
    else:
      spec.append(True)
  return array(spec)

# truncate the data to only the unique instances :
data = find_unique_row(data)

# index where the class is positive (1) and negative (0) :
pos = data[:,-1] == classes[1]
neg = data[:,-1] == classes[0]

unq_col = find_unique_col(data[:,:-1])
unq_pos = find_unique_col(data[:,:-1][pos])
unq_neg = find_unique_col(data[:,:-1][neg])

wild_neg = get_wild(unq_col, unq_neg)
wild_pos = get_wild(unq_col, unq_pos)

spec = get_spec(unq_neg, unq_pos)

for i,D in enumerate(data[:,:-1][pos]):
  if i == 0: S = D
  else:
    for d,s in zip(D,S):
      inter = intersect1d([d], s)
      if len(inter) == 0: 
        S = vstack((S, D))
        break



