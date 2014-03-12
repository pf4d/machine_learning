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

  Returns the means and standard deviations as a tuple.
  
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

def gain(S,Sc,A):
  """
  Calculate the information gain from a set <S> with corresponding class <Sc> 
  across an attribute <A>.
  """
  S = S[:,A]
  g = entropy(Sc)
  V = unique(S)
  n = len(S)
  for v in V:
    Sv = Sc[S == v]
    m  = float(len(Sv))
    g -= m / n * entropy(Sv)
  return g

def most_common(S):
  counts = bincount(S)
  return argmax(counts)

class node(object):
  
  def __init__(self, label, attrib=None, children=None):
    """
    Node of a decision tree node for attribute <attrib> with most common 
    class <label> and children nodes <children>.
    If <children> == None, this is either a leaf or a single-node tree.
    """
    self.label    = label
    self.children = children
    self.attrib   = attrib
    self.values   = None

  def add_child(self, child, val):
    if self.children == None:
      self.children = array([child])
      self.values   = array([val])
    else:
      self.children = append(self.children, child)
      self.values   = append(self.values,   val)

def ID3(S,Sc,Sg,A):
  """
  <S> is the training set with corresponding class <Sc>, <Sg> is a matrix of 
  unique values for all training and test data, and <A> is a list of attributes 
  that may be tested by the learned decision tree.  
  Returns a decision tree that correctly classifies the given <S>.
  """
  classes = unique(Sc)                    # unique values
  m       = len(Sg)                       # number of attributes
  if len(classes) == 1:                   
    return node(classes[0])               # return leaf node
  elif len(A) == 1:                       
    label = most_common(Sc, A[0])         # find the most common class
    return node(label)                    # return leaf node
  else:                                   
    gain_a = zeros(m)                     # array of info gains
    for a in A:                           
      gain_a[a] = gain(S,Sc,a)            # find the info gain for each attrib
    a_max    = argmax(gain_a)             # attrib. with highest info gain
    label    = most_common(Sc)            # find the most common class
    root     = node(label, a_max)         # create a new node
    S_amax   = S[:,a_max]                 # column of S with attrib a_max
    for v in Sg[a_max]:                   
      S_v = S[S_amax == v,:]              # subset of S with value v for Amax
      if shape(S_v)[0] == 0:              
        Sc_v  = Sc[S_amax == v]           # classes for S_v
        root.add_child(node(label), v)    # add the leaf
      else:
        Sc_v = Sc[S_amax == v]            # classes for S_v
        A_v  = A.copy()                   # copy of classes for recursion
        del A_v[a_max]                    # remove the attribute
        child = ID3(S_v, Sc_v, Sg, A_v)   # recurse
        root.add_child(child, v)          # add branch
    return root

def classify_DT(test, tree):
  """
  Classify single test case <test> with decision tree <tree>.
  """
  nde = tree                              # starting node
  # percolate through the tree :
  while nde.children != None:
    tv = t[nde.attrib]                    # corresponding value to node attrib.
    # find the branch to explore next :
    for i,v in enumerate(nde.values):
      if tv == v:
        nde = nde.children[i]             # set the node to the next attrib.
  return nde

#===============================================================================
# perform classification :

# plot the data:
mus, sigs = plot_dist(train, train_class, 10, 'DT')

# form array of unique values in S :
Sg = []
for a in vstack((train, test)).T:
  Sg.append(unique(a))
Sg = array(Sg)

# discover the decision tree :
tree = ID3(train, train_class, Sg, attrib)

# evaluate the test set and calculate accuracy :
v_DT = []
for t,c in zip(test, test_class):
  nde = classify_DT(t, tree)
  v_DT.append(nde.label)

# print the result (here 90.0%) :
correct = test_class == v_DT
print "percentage correct : %.1f%%" % (100*sum(correct)/float(len(correct)))

# plot the results :
plot_results(test_class, v_DT, 'DT')



