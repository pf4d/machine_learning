# bayes_network.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the Bayes network method to classify a set of forest fire 
# data.

import csv
import sys

import matplotlib.gridspec as gridspec

from time  import time
from pylab import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
# collect the data :
# columns : Storms, BusTourGroup, Lightning, Campfire, Thunder, Class

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {0 : 0, 1 : 1}
attrib  = {0 : 'Storms', 
           1 : 'BusTourGroup',
           2 : 'Lightning', 
           3 : 'Campfire', 
           4 : 'Thunder'}

# dataset :
f    = open(dirc + 'forestFireData.csv', 'r')
data = loadtxt(f, dtype='str', delimiter=",", skiprows=1)

#===============================================================================
# functions used to solve and plot results :

def get_normal(x, mu, sigma):
  """ 
  Function which returns the normal value at <x> with mean <mu> and 
  standard deviation <sigma>.
  """
  return 1/(sigma * sqrt(2 * pi)) * exp(-(x - mu)**2 / (2* sigma**2))

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


def k_cross_validate(k, data, classify_ftn, classes): 
  """
  Perform <k> crossvalidation on data <data> with classification function
  <classify_ftn> and possible classes dictionary <classes>.
  """
  n   = shape(data)[0]         # number of samples
  k   = 10                     # number of cross-validations
  idx = range(n)               # array of indices
  shuffle(idx)                 # randomize indices
  
  d   = split(data[idx], k)    # partition the shuffled data into k units
  
  # iterate through each partition and determine the results :
  result = []                  # final result array
  for i in range(k):
    test  = d[i][:,:-1]                           # test values
    testc = d[i][:,-1]                            # test classes
    train = vstack(d[0:i] + d[i+1:])              # training set
   
    V_ce = classify_ftn(test, train, classes)     # classify
     
    correct = sum(testc == V_ce)                  # number correct
    result.append(correct/float(len(testc)))      # append percentange
  
  return result


def classify_bayes_network(test, train, classes):
  """
  Classify a set of test data <test> with corresponding training data <train> 
  and dictionary of two possible classes <classes>.  The positive index of 
  <classes> is 1, while the negative index is 0.
  """
  return array(V_bn)


#===============================================================================
# perform classification with k-fold cross-validation :

k      = 10
result = k_cross_validate(k, data, classify_cand_elim, classes) 

print "percent correct: %.1f%%" % (100*average(result))




