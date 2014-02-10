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
test_class  = loadtxt(test_f, dtype='int', delimiter=",", 
                      usecols=(4,), skiprows=1)

# train classes :
train_f.seek(0)
train_class = loadtxt(train_f, dtype='int', delimiter=",", 
                      usecols=(4,), skiprows=1)

apple  = where(train_class == 1)[0]
peach  = where(train_class == 2)[0]
orange = where(train_class == 3)[0]
lemon  = where(train_class == 4)[0]

na = len(apple)
np = len(peach)
no = len(orange)
nl = len(lemon)

lens = [na, np, no, nl]

def plot_dist(train, train_class, nbins, norm=True):
  fig   = figure(figsize=(10,8))
  #fig  = figure()
  mus   = zeros((4,4))
  sigs  = zeros((4,4))
  bi_a  = []
  ct_a  = []
  for i in range(4):
    ax    = fig.add_subplot(220 + i+1)
    mini  = train[:,i].min()
    maxi  = train[:,i].max()
    bi_i  = []
    ct_i  = []
    for j in range(4):
      wj   = where(train_class == j+1)[0]
      xj   = train[wj,i]
      muj  = mean(xj)
      sigj = std(xj)
      rngi = linspace(mini, maxi, 1000)
      lblh = classes[j+1] + ' (%i)' % (j+1)
      lbln = r'$\mathcal{N}(\mu_%i, \sigma_%i^2)$' % (j+1, j+1)
      ct, bins, ign = ax.hist(train[wj, i], label=lblh, alpha=0.7, 
                              normed=norm)
      ax.plot(rngi, get_normal(rngi, muj, sigj), linewidth=2, label=lbln)
      mus[j,i]  = muj
      sigs[j,i] = sigj
      bi_i.append(bins)
      ct_i.append(ct)
    bi_a.append(array(bi_i))
    ct_a.append(array(ct_i))
  
    ax.set_title(attrib[i])
    if i == 2:
      leg = ax.legend()
      leg.get_frame().set_alpha(0.5)
    ax.grid()
  show()
  return mus, sigs, array(bi_a), array(ct_a)

mus, sigs, bins, cts = plot_dist(train, train_class, 10, norm=False)

i = 1
for mu, sig in zip(mus, sigs):
  if i == 4:  nsamp = 1000
  else:       nsamp = 10000
  n      = len(mu)
  b_samp = mu*ones((nsamp,n)) + sig*randn(nsamp,n)
  if i == 1:
    train_n       = b_samp
    train_class_n = i*ones(nsamp)
  else:
    train_n       = vstack((train_n, b_samp))
    train_class_n = append(train_class_n, i*ones(nsamp))
  i += 1

mus_n, sigs_n, bins_n, cts_n = plot_dist(train_n, train_class_n, 10, norm=True)


#===============================================================================
def find_ct_index(bins, v):
  idx = argmin(abs(bins - v))
  if bins[idx] < v: cti = idx
  else:             cti = idx - 1
  return cti

def classify_NB(test, test_class, bins, cts):
  P_test = []
  v_NB_a = []
  for va in test:
    P_mat = zeros((4,4))
    for i,v in enumerate(va):
      for j in range(4):
        if v > bins[i,j].max() or v < bins[i,j].min():
          P_mat[j,i] = 0.0
        else:
          k          = find_ct_index(bins[i,j], v)
          P_ij       = cts[i,j,k]
          P_mat[j,i] = P_ij
    sum_i   = sum(P_mat,  axis=0)
    P_mat  /= sum_i
    prod_j  = prod(P_mat, axis=1)
    v_NB    = argmax(prod_j) + 1
    v_NB_a.append(v_NB)
    P_test.append(P_mat)
  v_NB_a = array(v_NB_a)
  P_test = array(P_test)

  num_corr = sum(test_class - v_NB_a == 0)
  n        = float(len(test_class))
  print "Percent correct: %.1f%%" % (100 * num_corr / n)
  return v_NB_a

v_NB   = classify_NB(test, test_class, bins,   cts)
v_NB_n = classify_NB(test, test_class, bins_n, cts_n)

def plot_results(test_class, v_NB):
  fig = figure(figsize=(12,4))
  plot(abs(test_class - v_NB), drawstyle='steps-mid', lw=2.0)
  for x,c in zip(range(len(test_class)), test_class):
    if c == 4: 
      text(x, 0.0, c, horizontalalignment='center',
                      verticalalignment='center')
  tight_layout()
  title(r'$|v - v_{NB}|$')
  xlabel(r'$n$')
  ylim([-1, 2])
  grid()
  show()

plot_results(test_class, v_NB)
plot_results(test_class, v_NB_n)


