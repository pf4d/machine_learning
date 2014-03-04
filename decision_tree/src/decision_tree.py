# decision_tree.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the decision tree method to classify a set of tumor data.  
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
# columns : Clump Thickness, Uniformity of Cell Size, Uniformity of Cell Shape,
#           Marginal Adhesion, Single Epithelial Cell Size, Bare Nuclei, 
#           Bland Chromatin, Normal Nucleoli, Mitoses, class

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {0 : 'Negative', 1 : 'Positive'}
attrib  = {0 : 'Clump Thickness', 
           1 : 'Uniformity of Cell Size',
           2 : 'Uniformity of Cell Shape', 
           3 : 'Marginal Adhesion', 
           4 : 'Single Epithelial Cell Size', 
           5 : 'Bare Nuclei', 
           6 : 'Bland Chromatin', 
           7 : 'Normal Nucleoli', 
           8 : 'Mitoses'}

# training set :
train_f = open(dirc + 'train.csv', 'r')
train   = loadtxt(train_f, delimiter=",", usecols=(0,1,2,3,4,5,6,7,8), 
                  skiprows=1)
tr_min  = train.min(axis=0)      # array of mins
tr_max  = train.max(axis=0)      # array of maxs

# testing set :
test_f  = open(dirc + 'test.csv', 'r')
test    = loadtxt(test_f,  delimiter=",", usecols=(0,1,2,3,4,5,6,7,8), 
                  skiprows=1)

# test classes :
test_f.seek(0)
test_class  = loadtxt(test_f, dtype='int', delimiter=",", 
                      usecols=(9,), skiprows=1)

# train classes :
train_f.seek(0)
train_class = loadtxt(train_f, dtype='int', delimiter=",", 
                      usecols=(9,), skiprows=1)


#===============================================================================
# functions used to solve :

def get_normal(x, mu, sigma):
  """ 
  Function which returns the normal value at <x> with mean <mu> and 
  standard deviation <sigma>.
  """
  return 1/(sigma * sqrt(2 * pi)) * exp(-(x - mu)**2 / (2* sigma**2))

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
  mus   = zeros((m,n))                     # means matrix
  sigs  = zeros((m,n))                     # standard deviations matrix
  # iterate over each attribute i :
  for i in range(n):
    axi   = 100*ceil(sqrt(n)) + 10*sqrt(n)
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
                              normed=norm)
      # plot the results :
      ax.plot(rngi, get_normal(rngi, muj, sigj), linewidth=2, label=lbln)
      mus[j,i]  = muj                      # set the value of the mean
      sigs[j,i] = sigj                     # set the value of the std. dev. 
  
    ax.set_title(attrib[i])                # set the title
    if i == 2:
      leg = ax.legend()                    # add the legend
      leg.get_frame().set_alpha(0.5)       # transparent legend
    ax.grid()                              # gridlines
  tight_layout() 
  #savefig('../doc/images/' + name, dpi=300)
  show()
  return mus, sigs

def plot_results(test_class, v, name):
  """
  plot the absolute value of the differece between the true class <test_class>
  and classified class <v_NB>.
  """
  fig = figure(figsize=(12,4))
  plot(abs(test_class - v), 'k', drawstyle='steps-mid', lw=2.0, label='binned')
  for x,c in zip(range(len(test_class)), test_class):
    if c == 4: 
      text(x, 0.0, c, horizontalalignment='center',
                      verticalalignment='center')
  tight_layout()
  title(r'$|v - v_{NB}|$')
  xlabel(r'$n$')
  ylim([-1, 2])
  grid()
  legend()
  #savefig('../doc/images/' + name + '_results.png', dpi=300)
  show()

def entropy(S):
  """
  Calculate the entropy of a collection <S>.
  """
  E = 0
  n = len(S)
  for si in unique(S):
    c  = S[S == si]
    m  = float(len(c))
    pi = m/n
    E -= pi*log2(pi)
  return E

def gain(S,A):
  """
  Calculate the information gain from a set <S> across an attribute <A>.
  """
  S = S[:,A]
  g = entropy(S)
  V = unique(S)
  n = len(S)
  for v in V:
    Sv = S[S == v]
    m  = float(len(Sv))
    g -= m / n * entropy(Sv)
  return g

def most_common(S):
  counts = bincont(S)
  return argmax(counts)

class node(object):
  
  def __init__(self, label, children=None):
  """
  Node of a decision tree with label <label> and children nodes <children>.  
  If <children> == None, this is either a leaf or a single-node tree.
  """
    self.label    = label
    self.children = children

  def add_child(self, child):
    if self.children == None:
      self.children == array([child])
    else:
      self.children = append(self.children, child)

def ID3(S,Sc,A):
  """
  <S> is the training set with corresponding class <Sc>, <A> is a list of 
  attributes that may be tested by the learned decision tree.  
  Returns a decision tree that correctly classifies the given <S>.
  """
  classes = unique(Sc)                     # unique values
  if len(classes) == 1:
    return node(classes[0])                # return single-node tree
  elif len(A) == 1:
    label = most_common(Sc)                # find the most common class
    return node(label)                     # return single-node tree
  else:
    gain_a = array([])                     # array of info gains
    for a in A:
      gain_a = append(gain_a, gain(S,a))   # find the info gain for each attrib
    a_max    = argmax(gain_a)              # attrib. with highest info gain
    root     = node(a_max)                 # create a new node
    S_amax   = S[:,a_max]                  # column of S with attrib a_max
    for v in unique(S_amax):
      S_v = S[S_amax == v,:]               # subset of S with value v for Amax
      if shape(S_v)[0] == 0:
        Sc_v  = Sc[S_amax == v]            # classes for S_v
        label = most_common(Sc_v)          # find the most common class
        root.add_child(node(label))        # add the leaf
      else:
        Sc_v = Sc[S_amax == v]             # classes for S_v
        A_v  = A.copy()                    # copy of classes for recursion
        del A_v[a_max]                     # remove the attribute
        child = ID3(S_v, Sc_v, A_v)        # recurse
        root.add_child(child)              # add branch
    return root


#===============================================================================
# perform classification :


