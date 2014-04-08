# bayes_network.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the Bayes network method to classify a set of forest fire 
# data.

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
# columns : Storms, BusTourGroup, Lightning, Campfire, Thunder, Class

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {0 : '0', 1 : '1'}
attrib  = {0 : 'Storms', 
           1 : 'BusTourGroup',
           2 : 'Lightning', 
           3 : 'Campfire', 
           4 : 'Thunder'}

# dataset :
f    = open(dirc + 'forestFireData.csv', 'r')
data = loadtxt(f, dtype='int', delimiter=",", skiprows=1)

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


def cartesian(arrays, out=None):
    """
    Source: http://stackoverflow.com/questions/1208118/using-numpy-to-build-an-
            array-of-all-combinations-of-two-arrays
    
    Generate a cartesian product of input arrays.

    Parameters
    ----------
    arrays : list of array-like
        1-D arrays to form the cartesian product of.
    out : ndarray
        Array to place the cartesian product in.

    Returns
    -------
    out : ndarray
        2-D array of shape (M, len(arrays)) containing cartesian products
        formed of input arrays.

    Examples
    --------
    >>> cartesian(([1, 2, 3], [4, 5], [6, 7]))
    array([[1, 4, 6],
           [1, 4, 7],
           [1, 5, 6],
           [1, 5, 7],
           [2, 4, 6],
           [2, 4, 7],
           [2, 5, 6],
           [2, 5, 7],
           [3, 4, 6],
           [3, 4, 7],
           [3, 5, 6],
           [3, 5, 7]])

    """

    arrays = [np.asarray(x) for x in arrays]
    dtype = arrays[0].dtype

    n = np.prod([x.size for x in arrays])
    if out is None:
        out = np.zeros([n, len(arrays)], dtype=dtype)

    m = n / arrays[0].size
    out[:,0] = np.repeat(arrays[0], m)
    if arrays[1:]:
        cartesian(arrays[1:], out=out[0:m,1:])
        for j in xrange(1, arrays[0].size):
            out[j*m:(j+1)*m,1:] = out[0:m,1:]
    return out


def K2(attrib, max_parents, data):
  """
  """
  n = len(attrib)
  for i,a in zip(attrib, range(1,n+1)):
    predi = []
    Pold  = g(i, predi, data)
    proc  = True
    while proc and len(predi) < max_parents:
      f = []
      for node in attrib.keys():
        f.append(g(i, predi + [node], data))
      z = argmax(f)
      Pnew = g(i, predi + [z], data)
      if Pnew > Pold:
        Pold = Pnew
        predi = predi + [z]
      else:
        proc = False
    print 'Node :', attrib[i], '\tParent of', attrib[i], ':', predi


def g(i, predi, data):
  """
  """
  xi    = data[:,i]
  Vi    = unique(xi)
  ri    = len(Vi)
  d     = data[:,predi]
  m,n   = shape(d)

  def calc(N_ij, phi_i, qi):
    res   = 1
    for j in range(qi):
      alpha_ij = array([])
      for k in range(ri):
        idx       = where(xi == Vi[k])
        alpha_ijk = sum(d[idx] == phi_i[j])
        alpha_ij  = append(alpha_ij, alpha_ijk)
      eta = fact(ri - 1) / fact(N_ij + ri - 1) * prod(fact(alpha_ij))
      res += eta
    return res

  if len(predi) == 0:
    N_ij = len(data)
    res  = calc(N_ij, data, ri)
  
  else:
    unq   = []
    for j in range(n):
      unq.append(unique(d[:,j]))
    phi_i = cartesian(unq)
    qi    = len(phi_i)
    
    N_ij = sum(d == phi_i[j])
    res  = calc(N_ij, phi_i, qi)
  
  return res




def classify_bayes_network(test, train, classes):
  """
  Classify a set of test data <test> with corresponding training data <train> 
  and dictionary of two possible classes <classes>.  The positive index of 
  <classes> is 1, while the negative index is 0.
  """
  return array(V_bn)


#===============================================================================
# form test database :
attrib  = {0 : 'x1', 
           1 : 'x2',
           2 : 'x3'} 
D = array([[1,0,0],
           [1,1,1],
           [0,0,1],
           [1,1,1],
           [0,0,0],
           [0,1,1],
           [1,1,1],
           [0,0,0],
           [1,1,1],
           [0,0,0]])

K2(attrib, 2, D)

#===============================================================================
# perform classification with k-fold cross-validation :

k      = 10
#result = k_cross_validate(k, data, classify_cand_elim, classes) 

#print "percent correct: %.1f%%" % (100*average(result))




