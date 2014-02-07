# naive_bayes.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the naive Bayes method to classify a set of fruit data.  The
# accuracy is reported along with the time to compute.
#

import csv
import sys

import matplotlib.gridspec as gridspec

from time  import time
from pylab import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

def get_normal(x, mu, sigma):
  return 1/(sigma * sqrt(2 * pi)) * exp(-(x - mu)**2 / (2* sigma**2))

#===============================================================================
# collect the data :
# columns : Redness, Yellowness, Mass, Volume, Class

dirc  = '../data/'

# classes "lookup table" :
classes = {1 : 'apple',   2 : 'peach',      3 : 'orange', 4 : 'lemon'}
attrib  = {0 : 'redness', 1 : 'yellowness', 2 : 'mass',   3 : 'volume'}

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
test_class  = loadtxt(test_f, delimiter=",", usecols=(4,), skiprows=1)

# train classes :
train_f.seek(0)
train_class = loadtxt(train_f, delimiter=",", usecols=(4,), skiprows=1)

apple  = where(train_class == 1)[0]
peach  = where(train_class == 2)[0]
orange = where(train_class == 3)[0]
lemon  = where(train_class == 4)[0]

na = len(apple)
np = len(peach)
no = len(orange)
nl = len(lemon)

lens = [na, np, no, nl]

def plot_dist(train, train_class, norm=True):
  #fig  = figure(figsize=(18,12))
  fig  = figure()
  mus  = zeros((4,4))
  sigs = zeros((4,4))
  for i in range(4):
    ax    = fig.add_subplot(220 + i+1)
    mini  = train[:,i].min()
    maxi  = train[:,i].max()
    for j in range(4):
      wj   = where(train_class == j+1)[0]
      xj   = train[wj,i]
      muj  = mean(xj)
      sigj = std(xj)
      rngi = linspace(mini, maxi, 1000)
      lblh = classes[j+1] + ' (%i)' % (j+1)
      lbln = r'$\mathcal{N}(\mu_%i, \sigma_%i^2)$' % (j+1, j+1)
      ct, bins, ign = ax.hist(train[wj, i], label=lblh, alpha=0.7, normed=norm)
      ax.plot(rngi, get_normal(rngi, muj, sigj), linewidth=2, label=lbln)
      mus[j,i]  = muj
      sigs[j,i] = sigj
  
    ax.set_title(attrib[i])
    if i == 2:
      leg = ax.legend()
      leg.get_frame().set_alpha(0.5)
    ax.grid()
  show()
  return mus, sigs

ion()
mus, sigs = plot_dist(train, train_class, norm=True)

i = 1
for mu, sig in zip(mus, sigs):
  if i == 4:  nsamp = 1000
  else:       nsamp = 10000
  n      = len(mu)
  b_samp = mu*ones((nsamp,n)) + sig*randn(nsamp,n)
  if i == 1:
    train_new       = b_samp
    train_class_new = i*ones(nsamp)
  else:
    train_new = vstack((train_new, b_samp))
    train_class_new = append(train_class_new, i*ones(nsamp))
  i += 1

mus_new, sigs_new = plot_dist(train_new, train_class_new, norm=True)


##===============================================================================
## find the nearest neighbors :
#def calc_dist(si, sj):
#  dist = sqrt(sum((si - sj)**2))
#  return dist
#
## calculate the distance from each test case to each training case :
#dist = []
#for i, si in enumerate(test):
#  dist_i = []
#  stack  = vstack((si, train))   # stack test_i with the training set
#  tr_min = stack.min(axis=0)     # find min
#  tr_max = stack.max(axis=0)     # find max
#  for j, sj in enumerate(train):
#    si_norm = (si - tr_min) / (tr_max - tr_min) # normalize the fields
#    sj_norm = (sj - tr_min) / (tr_max - tr_min) # normalize the fields
#    dist_i.append(calc_dist(si_norm, sj_norm))
#  dist.append(array(dist_i))
#dist = array(dist)
#
## find the indicies of the sorted distances :
#ii   = argsort(dist)
#
#
##===============================================================================
## function for classifying :
#def classify(k):
#  #k  = int(sys.argv[1])
#  knn   = ii[:,:k]              # knn for each test fruit
#  res_i = []                    # array of votes
#  # add the weighted vote to an array of neighbor classes :
#  for i,j in enumerate(knn):
#    res_j = zeros(4)
#    tc    = train_class[j]      # class of neighbor j
#    dc    = dist[i][j]          # distance from test fruit i to neighbor j
#    # for each neighbor, add the weighted vote to an array of votes :
#    for c,d in zip(tc,dc):
#      i = classes[c]            # class index of neighbor
#      if weigh :
#        res_j[i] += 1/d**2      # add the inverse of square distance
#      else :
#        res_j[i] += 1           # add the non-weighted vote 
#    res_i.append(res_j)         # add the vote for test fruit i to the array
#  res_i = array(res_i)
#  
#  # calculate the guess class for all test classes simultaneously :
#  guess_class = argmax(res_i, axis=1)
#  return guess_class
#
#
##===============================================================================
## guess the class of the new fruit based on k nearest neighbors :
## for 0 < k <= m
#n = len(train_class)            # size of training set
#m = len(test_class)             # size of test set
#percent_correct = []            # percent correct for each k
#compute_time    = []            # time to compute for each k
#k = int(sys.argv[2])            # input k
#if k == 0: rng = range(1,n+1)   # if k == 0, compute for all k
#else :     rng = [k]            # else just do it once.
#for i in rng:
#  t0          = time()          # beginning time
#  guess_class = classify(i)     # classify with k
#  tf          = time()          # end time
#  
#  # calculate and display the percent correct and time to calculate :
#  correct = 0.0
#  for tc,gc in zip(test_class, guess_class):
#    if tc == inv_classes[gc]: correct += 1
#  per_cor = 100 * correct / m
#  t_tot   = tf - t0 
#  percent_correct.append(per_cor)
#  compute_time.append(t_tot)
#  print "Percent correct for k=%i : %.1f%% >>> Time to compute : %.3e seconds" \
#        % (i, per_cor, t_tot)
#
#
##===============================================================================
## plot the results if all k are computed :
#if k == 0:
#  if weigh: tit = 'fruit classification weighted by distance'
#  else:     tit = 'fruit classification without weighting'
#  pur = '#6e009d'
#  fig = figure()
#  ax1 = fig.add_subplot(111)
#  ax2 = ax1.twinx()
#  for tl in ax1.get_yticklabels():
#    tl.set_color('k')
#  for tl in ax2.get_yticklabels():
#    tl.set_color(pur)
#  
#  ax1.plot(range(1,n+1), percent_correct, color='k', 
#           drawstyle='steps-mid', lw=2.0) 
#  ax2.plot(range(1,n+1), compute_time,    color=pur, 
#           drawstyle='steps-mid', lw=2.0) 
#  ax1.set_xlabel(r'k')
#  ax1.set_ylabel(r'% correct',        color='k')
#  ax2.set_ylabel(r'compute time [s]', color=pur)
#  ax1.set_title(tit)
#  ax1.grid()
#  show()



