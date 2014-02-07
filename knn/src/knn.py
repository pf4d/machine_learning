# knn.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the knn method to classify a set of fruit data.  The
# accuracy is reported along with the time to compute.
#
# The first argument should be either 0 or 1 and determines whether weighted
# voting should be used.  The second argument determines k.  If this value is 
# 0, a loop will run and calculate for all 0 < k <= n, where n is the number of
# training data.

import csv
import sys

import matplotlib.gridspec as gridspec

from time  import time
from pylab import *

#===============================================================================
# collect the data :
# columns : Redness,Yellowness,Mass,Volume,Class
weigh = bool(int(sys.argv[1]))  # weigh or not

dirc  = '../data/'

# classes "lookup table" :
classes     = {'peach' : 0, 'orange' : 1, 'apple' : 2, 'lemon' : 3}
inv_classes = {0 : 'peach', 1 : 'orange', 2 : 'apple', 3 : 'lemon'}

# training set :
train_f = open(dirc + 'fruit.csv', 'r')
train   = loadtxt(train_f, delimiter=",", usecols=(0,1,2,3), skiprows=1)
tr_min  = train.min(axis=0)      # array of mins
tr_max  = train.max(axis=0)      # array of maxs

# testing set :
test_f  = open(dirc + 'testFruit.csv', 'r')
test    = loadtxt(test_f,  delimiter=",", usecols=(0,1,2,3), skiprows=1)

# test classes :
test_f.seek(0)
test_class  = loadtxt(test_f, dtype='str', delimiter=",", usecols=(4,), 
                      skiprows=1)

# train classes :
train_f.seek(0)
train_class = loadtxt(train_f, dtype='str', delimiter=",", usecols=(4,), 
                      skiprows=1)


#===============================================================================
# find the nearest neighbors :
def calc_dist(si, sj):
  dist = sqrt(sum((si - sj)**2))
  return dist

# calculate the distance from each test case to each training case :
dist = []
for i, si in enumerate(test):
  dist_i = []
  stack  = vstack((si, train))   # stack test_i with the training set
  tr_min = stack.min(axis=0)     # find min
  tr_max = stack.max(axis=0)     # find max
  for j, sj in enumerate(train):
    si_norm = (si - tr_min) / (tr_max - tr_min) # normalize the fields
    sj_norm = (sj - tr_min) / (tr_max - tr_min) # normalize the fields
    dist_i.append(calc_dist(si_norm, sj_norm))
  dist.append(array(dist_i))
dist = array(dist)

# find the indicies of the sorted distances :
ii   = argsort(dist)


#===============================================================================
# function for classifying :
def classify(k):
  #k  = int(sys.argv[1])
  knn   = ii[:,:k]              # knn for each test fruit
  res_i = []                    # array of votes
  # add the weighted vote to an array of neighbor classes :
  for i,j in enumerate(knn):
    res_j = zeros(4)
    tc    = train_class[j]      # class of neighbor j
    dc    = dist[i][j]          # distance from test fruit i to neighbor j
    # for each neighbor, add the weighted vote to an array of votes :
    for c,d in zip(tc,dc):
      i = classes[c]            # class index of neighbor
      if weigh :
        res_j[i] += 1/d**2      # add the inverse of square distance
      else :
        res_j[i] += 1           # add the non-weighted vote 
    res_i.append(res_j)         # add the vote for test fruit i to the array
  res_i = array(res_i)
  
  # calculate the guess class for all test classes simultaneously :
  guess_class = argmax(res_i, axis=1)
  return guess_class


#===============================================================================
# guess the class of the new fruit based on k nearest neighbors :
# for 0 < k <= m
n = len(train_class)            # size of training set
m = len(test_class)             # size of test set
percent_correct = []            # percent correct for each k
compute_time    = []            # time to compute for each k
k = int(sys.argv[2])            # input k
if k == 0: rng = range(1,n+1)   # if k == 0, compute for all k
else :     rng = [k]            # else just do it once.
for i in rng:
  t0          = time()          # beginning time
  guess_class = classify(i)     # classify with k
  tf          = time()          # end time
  
  # calculate and display the percent correct and time to calculate :
  correct = 0.0
  for tc,gc in zip(test_class, guess_class):
    if tc == inv_classes[gc]: correct += 1
  per_cor = 100 * correct / m
  t_tot   = tf - t0 
  percent_correct.append(per_cor)
  compute_time.append(t_tot)
  print "Percent correct for k=%i : %.1f%% >>> Time to compute : %.3e seconds" \
        % (i, per_cor, t_tot)


#===============================================================================
# plot the results if all k are computed :
if k == 0:
  if weigh: tit = 'fruit classification weighted by distance'
  else:     tit = 'fruit classification without weighting'
  pur = '#6e009d'
  fig = figure()
  ax1 = fig.add_subplot(111)
  ax2 = ax1.twinx()
  for tl in ax1.get_yticklabels():
    tl.set_color('k')
  for tl in ax2.get_yticklabels():
    tl.set_color(pur)
  
  ax1.plot(range(1,n+1), percent_correct, color='k', 
           drawstyle='steps-mid', lw=2.0) 
  ax2.plot(range(1,n+1), compute_time,    color=pur, 
           drawstyle='steps-mid', lw=2.0) 
  ax1.set_xlabel(r'k')
  ax1.set_ylabel(r'% correct',        color='k')
  ax2.set_ylabel(r'compute time [s]', color=pur)
  ax1.set_title(tit)
  ax1.grid()
  show()



