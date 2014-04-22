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


def K2(nodes, attrib, max_parents, data, g_ftn):
  """
  K2 algorithm acts on a set of nodes <nodes> with corresponding real value 
  dictionary <attrib>.  <max_parents> sets the max number of parents for each 
  node, <data> is the data to operate with, and <g_ftn> either function g or
  log_g.
  """
  n     = len(nodes)                                 # number of parameters
  par_a = []                                         # main parent array

  # iterate through each node (attribute) :
  for i,a in zip(nodes, range(1,n+1)):
    par_i = array([], 'int')                         # parents for node i
    Pold  = g_ftn(i, par_i, data)                    # rate the node
    proc  = True                                     # ok to proceed?
    predi = nodes[:i]                                # predicessors of node i
    
    # while ok to proceed and parents are less than the max :
    while proc and len(par_i) < max_parents:
      seti = setdiff1d(predi, par_i)                 # predi - par_i
      f    = []                                      # array of parental prob.
      # iterate through the set predi - par_i
      for s in seti:
        f.append(g_ftn(i, append(par_i, s), data))   # append prob. to f
      # if predi == par_i :
      if len(seti) == 0:
        z  = array([], 'int')                        # parents are none 
      else:
        z  = seti[argmax(f)]                         # parents are max f
      Pnew = g_ftn(i, append(par_i, z), data)        # new probability
      # if things get better :
      if Pnew > Pold:
        Pold  = Pnew                                 # replace probabiltity
        par_i = append(par_i, z)                     # add new parent to par_i
      else:
        proc = False                                 # no longer proceed
    
    parents = []                                     # form parents array 
    for p in par_i:
      parents.append(attrib[p])
    par_a.append(par_i)                              # add par_i to par_a
    print 'Node :', attrib[i], '\tParent of', attrib[i], ':', parents

  return array(par_a)


def g(i, par_i, data):
  """
  returns the probability that node (attribute) <i> of data matrix <data> has
  the parents array <par_i>.
  """
  xi    = data[:,i]          # data column i
  Vi    = unique(xi)         # unique values in data column i
  ri    = len(Vi)            # number of unique values in data column i
  d     = data[:,par_i]      # data truncated to only the parent nodes
  m,n   = shape(d)           # size of truncated data
  z     = len(par_i)         # number of parent nodes in par_i
  
  # if no parents, return the naive bayes probability :
  if z == 0:
    N_ij     = len(data)
    alpha_i_ = array([])
    for k in range(ri):
      alpha_i_k = sum(xi == Vi[k])
      alpha_i_  = append(alpha_i_, alpha_i_k)
    res = fact(ri - 1) / fact(N_ij + ri - 1) * prod(fact(alpha_i_))
  
  # otherwise include conditional probability terms :
  else:
    unq   = []                      # unique parent values list
    # iterate through each dimenson :
    for j in range(n):
      unq.append(unique(d[:,j]))    # append the unique values of parent j
    phi_i = cartesian(unq)          # form the cross product of parental values
    qi    = len(phi_i)              # number of possible parental combinations
    
    res   = 1
    for j in range(qi):
      N_ij     = sum((d == phi_i[j]).all(axis=1))
      alpha_ij = array([])
      for k in range(ri):
        idx       = where(xi == Vi[k])
        alpha_ijk = sum((d[idx] == phi_i[j]).all(axis=1))
        alpha_ij  = append(alpha_ij, alpha_ijk)
      eta  = fact(ri - 1) / fact(N_ij + ri - 1) * prod(fact(alpha_ij))
      res *= eta
  
  return res


def logFact(n):
  """
  take the log factorial of a numpy array or integer <n>.
  """
  if type(n) == ndarray:
    res = []
    for i in n:
      res.append(sum(log(arange(1,i+1))))
  elif type(n) == int64 or type(n) == int:
    if n == 0:
      res = 0
    else:
      res = sum(log(arange(1,n+1)))
  else:
    print type(n), "NOT SUPPORTED BY LOGFACT"
  return res


def log_g(i, par_i, data):
  """
  returns the probability that node (attribute) <i> of data matrix <data> has
  the parents array <par_i> using log factorial to get around the factorial 
  barrier.
  """
  xi    = data[:,i]          # data column i
  Vi    = unique(xi)         # unique values in data column i
  ri    = len(Vi)            # number of unique values in data column i
  d     = data[:,par_i]      # data truncated to only the parent nodes
  m,n   = shape(d)           # size of truncated data
  z     = len(par_i)         # number of parent nodes in par_i

  # if no parents, return the naive bayes probability :
  if z == 0:
    N_ij     = len(data)
    alpha_i_ = array([])
    for k in range(ri):
      alpha_i_k = sum(xi == Vi[k])
      alpha_i_  = append(alpha_i_, alpha_i_k)
    res = logFact(ri - 1) - logFact(N_ij + ri - 1) + sum(logFact(alpha_i_))
  
  # otherwise include conditional probability terms :
  else:
    unq   = []                      # unique parent values list
    # iterate through each dimenson :
    for j in range(n):
      unq.append(unique(d[:,j]))    # append the unique values of parent j
    phi_i = cartesian(unq)          # form the cross product of parental values
    qi    = len(phi_i)              # number of possible parental combinations
    
    res   = 0
    for j in range(qi):
      N_ij     = sum((d == phi_i[j]).all(axis=1))
      alpha_ij = array([])          # array of alpha_ijk's
      for k in range(ri):
        idx       = where(xi == Vi[k])
        alpha_ijk = sum((d[idx] == phi_i[j]).all(axis=1))
        alpha_ij  = append(alpha_ij, alpha_ijk)
      eta  = logFact(ri - 1) - logFact(N_ij + ri - 1) + sum(logFact(alpha_ij))
      res += eta
  
  return res


def classify_bayes_network(test, train, classes, params):
  """
  Classify a set of test data <test> with corresponding training data <train> 
  and dictionary of two possible classes <classes>.  The positive index of 
  <classes> is 1, while the negative index is 0.
  """
  attrib  = params[0]                                  # attribute dictionary
  max_par = params[1]                                  # max number of parents
  g_ftn   = params[2]                                  # g function to use
                                                       
  nodes   = sort(attrib.keys())                        # nodes 
  network = K2(nodes, attrib, max_par, train, g_ftn)   # B.B.S.
  
  V_bn = []                             # classified classes
  # iterate through each test instance and classify :
  for t in test:
    # for each dimension in t
    for i,xi in enumerate(t):
      pars = network[i]                       # form the DAG
      # if no parents :
      if len(pars) == 0:
        votes = []                            # array of votes
        # for each possible class :
        for c in classes.keys():
          cnt = sum(train[:,-1] == c)         # count up the number
          votes.append(cnt)                   # add to tally
        vi = argmax(votes)                    # find the max
      # otherwise, include conditional probability terms :
      else:
        votes = []                            # array of votes
        # for each possible class :
        for c in classes.keys():
          d = train.copy()                    # copy the training data
          # for each parent :
          for p in pars:
            idx = where(d[:,p] == c)          # index where class matches
            d   = d[idx]                      # truncate the data
          votes.append(len(d))                # vote is the number left
        vi = argmax(votes)                    # find the max
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
attrib_n  = {0 : 'x1', 
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

max_par = inf
#network = K2(sort(attrib_n.keys()), attrib_n, max_par, D, g)
#network = K2(sort(attrib.keys()), attrib, max_par, data, log_g)

#===============================================================================
# perform classification with k-fold cross-validation :

k       = 10
max_par = inf
g_ftn   = log_g
params  = [attrib, max_par, log_g]
result  = k_cross_validate(k, data, classify_bayes_network, classes, params) 

print "\npercent correct: %.1f%%" % (100*average(result))




