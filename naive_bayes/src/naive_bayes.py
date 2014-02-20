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
  """ 
  Function which returns the normal value at <x> with mean <mu> and 
  standard deviation <sigma>.
  """
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

# total number of observations :
m  = len(train_class)
na = len(apple)
np = len(peach)
no = len(orange)
nl = len(lemon)

lens = [na, np, no, nl]


#===============================================================================
def plot_dist(train, train_class, nbins, name, norm=True):
  """
  Function which plots a histogram of data <train> with associated training 
  class <train_class> with number of bins <nbins>, title <name>.  If <norm> 
  is True, the bins are normalized.

  Returns the means, standard deviations, bins, and bin counts as a tuple.
  
  Image is saved in directory ../doc/images/.
  """
  vs    = unique(train_class)              # different values
  m     = len(vs)                          # number of distinct values
  n     = shape(train)[1]                  # number of different attributes
  fig   = figure(figsize=(10,8))           # figure instance
  mus   = zeros((4,4))                     # means matrix
  sigs  = zeros((4,4))                     # standard deviations matrix
  bi_a  = []                               # array of bins
  ct_a  = []                               # array of bin counts
  # iterate over each attribute i :
  for i in range(n):
    ax    = fig.add_subplot(220 + i+1)     # create a subplot
    mini  = train[:,i].min()               # min of attribute i
    maxi  = train[:,i].max()               # max of attribute i
    bi_i  = []                             # bin array for attribute i
    ct_i  = []                             # bin count array for attribute i
    # iterate over each class j :
    for j in range(m):
      wj   = where(train_class == j+1)[0]  # indicies for class j
      xj   = train[wj,i]                   # array of training values j
      muj  = mean(xj)                      # mean of class j
      sigj = std(xj)                       # standard dev of class j
      rngi = linspace(mini, maxi, 1000)    # range of plotting normal curve
      lblh = classes[j+1] + ' (%i)' % (j+1)
      lbln = r'$\mathcal{N}(\mu_%i, \sigma_%i^2)$' % (j+1, j+1)
      # function returns the bin counts and bins
      ct, bins, ign = ax.hist(train[wj, i], nbins, label=lblh, alpha=0.7, 
                              normed=norm)
      # plot the results :
      ax.plot(rngi, get_normal(rngi, muj, sigj), linewidth=2, label=lbln)
      mus[j,i]  = muj                      # set the value of the mean
      sigs[j,i] = sigj                     # set the value of the std. dev. 
      bi_i.append(bins)                    # add the bin to the list
      ct_i.append(ct)                      # add the bin counts to the list
    bi_a.append(array(bi_i))               # convert to numpy array
    ct_a.append(array(ct_i))               # convert to numpy array
  
    ax.set_title(attrib[i])                # set the title
    if i == 2:
      leg = ax.legend()                    # add the legend
      leg.get_frame().set_alpha(0.5)       # transparent legend
    ax.grid()                              # gridlines
  tight_layout() 
  #savefig('../doc/images/' + name, dpi=300)
  show()
  return mus, sigs, array(bi_a), array(ct_a)

# get the non-normalized data :
nbins = 100
mus, sigs, bins, cts = plot_dist(train, train_class,  nbins, 
                                 'not_normed_original', norm=False)
# get the normalized data :
#mus, sigs, bins, cts = plot_dist(train, train_class, nbins,
#                                 'normed_original',    norm=True)

# develop a new training data with the same mean and standard deviation
# as the original but with far fewer lemon (class = 4) observations.
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

# get the non-normalized data :
mus_n, sigs_n, bins_n, cts_n = plot_dist(train_n, train_class_n, nbins,
                                         'not_normed_new', norm=False)
# get the normalized data :
#mus_n, sigs_n, bins_n, cts_n = plot_dist(train_n, train_class_n, nbins,
#                                         'normed_new',     norm=True)


#===============================================================================
def find_ct_index(bins, v):
  """
  Find the bin count index of value <v> in bin array <bins>.
  """
  idx = argmin(abs(bins - v))
  if bins[idx] < v: cti = idx
  else:             cti = idx - 1
  return cti

def classify_NB(test, test_class, train, train_class, bins, cts, 
                param=False, mu=None, sig=None):
  """
  Perform Naive Bays classification with test data array <test>, test class 
  array <test_class>, training data array <train>, training class array 
  <train_class>, bins <bins>, and bin counts <cts>.  If <param> is True, 
  use the parametric approach with means and standard deviation matrices 
  <mu> and <sig>.
  """
  P_test = []                     # Probability array
  v_NB_a = []                     # classified class array
  q      = len(train_class)       # total number of observations
  vs     = unique(test_class)     # different possible values
  m      = len(vs)                # number of different values
  n      = shape(train)[1]        # number of different attributes
  
  # find total number of each values in training set :
  lens   = []
  for v in vs:
    lens.append(sum(train_class == v))
  lens = array(lens)

  # the probability of getting value j from out training data :
  Pvj  = lens/float(q)
  
  # classify each test value va :
  for va in test:
    P_mat = zeros((m,n))     # probabilities initialized to zero
    # iterate through each attribute for test value va :
    for i,v in enumerate(va):
      # iterate over each different value :
      for j in range(m):
        if param:
          P_mat[j,i] = get_normal(v, mus[j,i], sigs[j,i])
        else:
          # if the value is outside of the bin range, probability is zero :
          if v > bins[i,j].max() or v < bins[i,j].min():
            P_mat[j,i] = 1.0
          # otherwise it is the value of the bin count :
          else:
            k          = find_ct_index(bins[i,j], v)
            P_ij       = cts[i,j,k] + 1.0
            P_mat[j,i] = P_ij
    sum_i   = sum(P_mat,  axis=0)  # sum over each class probability
    P_mat  /= (sum_i + lens)       # normalize the probabilities
    Pa_vj   = prod(P_mat, axis=1)  # find the product of probabilities
    prior   = Pvj * Pa_vj          # the prior
    v_NB    = argmax(prior) + 1    # find the maximum index of probability
    v_NB_a.append(v_NB)            # append the ML class to result list
    P_test.append(P_mat)           # append the probability matrix to list
  v_NB_a = array(v_NB_a)           # convert list to array
  P_test = array(P_test)           # convert list to array

  num_corr = sum(test_class - v_NB_a == 0)  # number correct
  n        = float(len(test_class))         # number of test classes
  print "Percent correct: %.1f%%" % (100 * num_corr / n)
  return v_NB_a

v_NB   = classify_NB(test, test_class, train, train_class, bins, cts) 
v_NB_p = classify_NB(test, test_class, train, train_class, bins, cts, 
                     True, mus, sigs)
v_NB_n = classify_NB(test, test_class, train, train_class, bins_n, cts_n)

def plot_results(test_class, v_NB):
  """
  plot the absolute value of the differece between the true class <test_class>
  and classified class <v_NB>.
  """
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

plot_results(test_class, v_NB)    # plot the results for original data
plot_results(test_class, v_NB_p)  # plot the results for parametric approach
plot_results(test_class, v_NB_n)  # plot the results for new data


