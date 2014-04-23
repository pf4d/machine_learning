# neural_network.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the neural network method to classify a set of iris data.

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
# columns : Sepal Length, Sepal Width, Petal Length, Petal Width, Species

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {1 : '1', 2 : '2', 3 : '3'}
attrib  = {0 : 'Sepal Length', 
           1 : 'Sepal Width',
           2 : 'Petal Length', 
           3 : 'Petal Width'} 

# dataset :
f    = open(dirc + 'iris.csv', 'r')
data = loadtxt(f, dtype='float', delimiter=",", skiprows=1)

#===============================================================================
# functions used to solve and plot results :

def get_normal(x, mu, sigma):
  """ 
  Function which returns the normal value at <x> with mean <mu> and 
  standard deviation <sigma>.
  """
  return 1/(sigma * sqrt(2 * pi)) * exp(-(x - mu)**2 / (2* sigma**2))


def plot_dist(train, nbins, name, rows, cols, plot_normal=False, 
              log=False, norm=True):
  """
  Function which plots a histogram of data <train> with associated training 
  class <train_class> with number of bins <nbins>, title <name>.  If <norm> 
  is True, the bins are normalized.

  Returns the means and standard deviations as a tuple.
  
  Image is saved in directory ../doc/images/.
  """
  train_class = train[:,-1]
  train       = train[:,:-1]
  vs    = unique(train_class)              # different values
  m     = len(vs)                          # number of distinct values
  n     = shape(train)[1]                  # number of different attributes
  fig   = figure(figsize=(10,8))           # figure instance
  mus   = zeros((m,n))                     # means matrix
  sigs  = zeros((m,n))                     # standard deviations matrix
  axi   = 100*rows + 10*cols               # format for number of plots
  # iterate over each attribute i :
  for i in range(n):
    ax    = fig.add_subplot(axi + i+1)     # create a subplot
    mini  = train[:,i].min()               # min of attribute i
    maxi  = train[:,i].max()               # max of attribute i
    ax.set_xlim([mini, maxi])
    # iterate over each class j :
    for j in range(m):
      wj   = where(train_class == j)[0]  # indicies for class j
      xj   = train[wj,i]                   # array of training values j
      muj  = mean(xj)                      # mean of class j
      sigj = std(xj)                       # standard dev of class j
      rngi = linspace(mini, maxi, 1000)    # range of plotting normal curve
      lblh = classes[j] + ' (%i)' % (j)
      lbln = r'$\mathcal{N}(\mu_%i, \sigma_%i)$' % (j, j)
      # function returns the bin counts and bins
      ct, bins, ign = ax.hist(train[wj, i], nbins, label=lblh, alpha=0.7, 
                              log=log, normed=norm)
      # plot the results :
      if plot_normal:
        ax.plot(rngi, get_normal(rngi, muj, sigj), linewidth=2, label=lbln)
      mus[j,i]  = muj                      # set the value of the mean
      sigs[j,i] = sigj                     # set the value of the std. dev. 
  
    ax.set_title(attrib[i])                # set the title
    if i == 2:
      leg = ax.legend()                    # add the legend
      leg.get_frame().set_alpha(0.5)       # transparent legend
    ax.grid()                              # gridlines
  tight_layout() 
  savefig('../doc/images/' + name, dpi=300)
  show()
  return mus, sigs


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

def sigmoid(v):
  return 1 / (1 + exp(-v))


class network(object):
  
  def __init__(self, train, trans_ftn, eta, n_neurons):
    """
    """
    self.train     = train[:,:-1]           # training data
    self.train_c   = train[:,-1]            # training class
    self.classes   = unique(self.train_c)   # possible classes
    self.trans_ftn = trans_ftn              # array of layer trans. ftns.
    self.eta       = eta                    # array of layer relaxation params.
    self.n_neurons = n_neurons              # array of layer num. of neurons
    self.n,self.m  = shape(self.train)      # num. of training data / dim's
    
    # create leyers and network structure : 
    network   = []
    for i, (n, t, e) in enumerate(zip(n_neurons, trans_ftn, eta)):
      # number of weights == num of dimensions :
      if i == 0: w = m
      # num of weights == num of neurons in prev. layer :
      else:      w = n_neurons[i-1]
      
      # create layer and add the layer to the network :
      layer = []
      for j in range(n):
        layer.append(neuron(w,t,e))
      network.append(array(layer))
    self.network = array(network)



class neuron(object):

  def __init__(self, n, trans_ftn, eta):
    """
    """
    self.n         = n                 # number of inputs
    self.trans_ftn = trans_ftn         # transfer function
    self.eta       = eta               # relaxation parameter
    self.w = 0.10*random(n+1) - 0.05   # init. n+1 weights from [-0.05, 0.05]


def classify_neural_network(test, train, classes, params):
  """
  Classify a set of test data <test> with corresponding training data <train> 
  and dictionary of two possible classes <classes>.  The positive index of 
  <classes> is 1, while the negative index is 0.
  """
  V_bn = []                             # classified classes
  # iterate through each test instance and classify :
  for t in test:
    V_bn.append(vi)                           # append result
  return array(V_bn)


def k_cross_validate(k, data, classify_ftn, classes, ftn_params=None): 
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
    print "\nFOLD %i" % (i+1)
    print "-----------------------------------------------------------"
    test  = d[i][:,:-1]                           # test values
    testc = d[i][:,-1]                            # test classes
    train = vstack(d[0:i] + d[i+1:])              # training set
   
    V_c = classify_ftn(test, train, classes, ftn_params)     # classify
     
    correct = sum(testc == V_c)                   # number correct
    result.append(correct/float(len(testc)))      # append percentange
  
  return array(result)


#===============================================================================
# form test database :

#===============================================================================
# perform classification with k-fold cross-validation :

k       = 10
#result  = k_cross_validate(k, data, classify_neural_network, classes) 

#print "\npercent correct: %.1f%%" % (100*average(result))




