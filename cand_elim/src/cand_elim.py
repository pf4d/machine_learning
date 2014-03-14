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

from itertools import permutations, combinations
from time      import time
from pylab     import *

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
  unq  = []
  lens = []
  for i in range(shape(S)[1]):
    t = unique(S[:,i])
    unq.append(t)
    lens.append(len(t))
  return array(unq), array(lens)

def find_unique_row(S):
  """
  find the unique values of each row of array <S>.
  """
  unq = unique(S.view(S.dtype.descr * S.shape[1]))
  return unq.view(S.dtype).reshape(-1, S.shape[1])

def get_gen_and_spec(unq, unq_pos, unq_neg):
  spc = []
  gen = []
  for up, un in zip(unq_pos, unq_neg):
    inter = intersect1d(up, un)
    # none of up intersect with un :
    if len(inter) == 0:
      spc.append(True)
    # up == un :
    elif len(inter) == len(up) and len(inter) == len(un):
      spc.append(True)
    # up != un :
    else:
      spc.append(False)
  spc = array(spc)
  for uc, up in zip(unq, unq_pos):
    if len(uc) == len(up): gen.append(False)
    else:                  gen.append(True)
  gen = array(gen) & spc
  return spc, gen

# test data from the book :
data1 = array(['Sunny','Warm','Normal','Strong','Warm','Same','Enjoy Sport'], 
              dtype='|S12')
data2 = array(['Sunny','Warm','High','Strong','Warm','Same','Enjoy Sport'], 
              dtype='|S12')
data3 = array(['Rainy','Cold','High','Strong','Warm','Change','Do Not Enjoy'], 
              dtype='|S12')
data4 = array(['Sunny','Warm','High','Strong','Cool','Change','Enjoy Sport'], 
              dtype='|S12')
data = vstack((data1, data2, data3, data4))

# truncate the data to only the unique instances :
data = find_unique_row(data)

# index where the class is positive (1) and negative (0) :
pos = data[:,-1] == classes[1]
neg = data[:,-1] == classes[0]

unq_col, col_len = find_unique_col(data[:,:-1])
unq_pos, pos_len = find_unique_col(data[:,:-1][pos])
unq_neg, neg_len = find_unique_col(data[:,:-1][neg])

spc, gen = get_gen_and_spec(unq_col, unq_pos, unq_neg)

n   = len(unq_pos)

t = where(spc)[0]
g = ((pos_len - neg_len) > 0) & gen
result = []
for k in range(2, len(t) + 1):
  print "\n %i wild :" % (n - k)
  print "--------------------------------------------------"
  for s in combinations(t, k):
    idx  = array(s)
    temp = zeros(len(unq_pos), dtype='S12')
    for i in idx:
      temp[i] = unq_pos[i][0]
    result.append(temp)
    print temp

t = where(gen)[0]
print "\n %i wild :" % (n - 1)
print "--------------------------------------------------"
for s in combinations(t, 1):
  idx  = array(s)
  temp = zeros(len(unq_pos), dtype='S12')
  for i in idx:
    temp[i] = unq_pos[i][0]
  result.append(temp)
  print temp
result  = array(result)

multval = unq_pos[g]
for mv,i in zip(multval, where(g==True)[0]):
  zi = where(result[:,i] == mv[0])[0]
  new = result[zi]
  for v in mv[1:]:
    new[:,i] = v

if len(multval) > 0: result = vstack((result, new))
print "\n\n final results with any multivalues added :"
print "--------------------------------------------------"
print result



