# cand_elim.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the candidate elimination method to classify a set of data.  
# The accuracy is reported along with the time to compute.
#

import csv
import sys

import matplotlib.gridspec as gridspec

from itertools import combinations, chain
from time      import time
from pylab     import *

mpl.rcParams['font.family']     = 'serif'
mpl.rcParams['legend.fontsize'] = 'medium'

#===============================================================================
# collect the data :
# columns : Sky, AirTemp, Humidity, Wind, Water, Forecast, class

dirc  = '../data/'       # directory containing data

# classes "lookup table" :
classes = {0 : 'Do Not Enjoy', 1 : 'Enjoy Sport'}
attrib  = {0 : 'Sky', 
           1 : 'Air Temp',
           2 : 'Humidity', 
           3 : 'Wind', 
           4 : 'Water', 
           5 : 'Forecast'}

# dataset :
f    = open(dirc + 'trainingDataCandElim.csv', 'r')
data = loadtxt(f, dtype='str', delimiter=",", skiprows=1)


#===============================================================================
# functions used to solve and plot results :

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

def find_unique_col(S):
  """
  find the unique values of each column of array <S>.
  """
  unq  = []
  lens = []
  for i in range(shape(S)[1]):
    t = unique(S[:,i])
    unq.append(t)
    lens.append(len(t))
  return array(unq), array(lens)

def find_unique_row(S):
  """
  find the unique values of each row of array <S>.
  """
  unq = unique(S.view(S.dtype.descr * S.shape[1]))
  return unq.view(S.dtype).reshape(-1, S.shape[1])

def candidate_elimination(data, classes):
  """
  Determine the version space of data array <data> with the last column equal 
  to one of the two choices in the dictionary <classes>.  Returns a tuple 
  containing the specific boundary array, general boundary array, and 
  the version space array in that order.
  """
  # index where the class is positive (1) and negative (0) :
  pos = data[:,-1] == classes[1]
  neg = data[:,-1] == classes[0]
  
  # get the unique values of the columns, positive vals, and negative vals :
  unq_col, col_len = find_unique_col(data[:,:-1])
  unq_pos, pos_len = find_unique_col(data[:,:-1][pos])
  unq_neg, neg_len = find_unique_col(data[:,:-1][neg])
  
  spc = []  # specific boundary
  gen = []  # general boundary
 
  if sum(pos_len) == 0:
    print "ERROR: THERE ARE NO POSITIVE VALUES - ALGORITHM EXITS"
    sys.exit(0)
  
  # iterate through all the unique values and determine the boundaries :
  for i, (up, un, pn, nn) in enumerate(zip(unq_pos, unq_neg, pos_len, neg_len)):
    inter = intersect1d(up, un)
    # none of up intersect with un :
    if len(inter) == 0:
      spc.append(True)
      gen.append(True)
    # there are more negative than positives :
    elif nn > pn:
      spc.append(True)
      gen.append(False)
    # there are more positives than negatives : 
    elif nn < pn:
      spc.append(False)
      gen.append(False)
    # all intersect :
    elif len(inter) == nn and nn == pn:
      # if there is only one choice of positive and negative value :
      if nn == 1 and pn == 1:
        spc.append(True)
      # if there is more than one choice of positive value :
      else:
        spc.append(False)
      gen.append(False)
  
  # convert to arrays :
  spc = array(spc)
  gen = array(gen)

  n  = len(unq_pos)                        # number of dimensions
  ts = where(spc)[0]                       # indicies of specific boundaries
  tg = where(gen)[0]                       # indicies of general boundaries
  g  = ((pos_len - neg_len) > 0) & gen     # where there are multiple pos. vals.
  vs = []                                  # final version space
  multval = unq_pos[g]                     # multiple positive values
  
  # add to the version space all combinations of the specific values, this
  # corresponds to the most specific boundary :
  for k in range(2, len(ts) + 1):
    print "\n %i wild :" % (n - k)
    print "--------------------------------------------------"
    # iterate through each combination of array indices :
    for idx in combinations(ts, k):
      temp = zeros(len(unq_pos), dtype='S12')    # new version sp. entity
      # for each index, put the correct value in the spot :
      for i in idx:
        temp[i] = unq_pos[i][0]
      vs.append(temp)
      print temp
  
  # add to the version space all combinations of the general values, this
  # corresponds to the most general boundary :
  print "\n %i wild :" % (n - 1)
  print "--------------------------------------------------"
  # iterate through each combination of array indices :
  for idx in combinations(tg, 1):
    temp = zeros(len(unq_pos), dtype='S12')    # new version sp. entity
    # for each index, put the correct value in the spot :
    for i in idx:
      temp[i] = unq_pos[i][0]
    vs.append(temp)
    print temp
  
  # convert version space array :
  vs  = array(vs)
  
  # add to the dataset all combinations of multiple positive values, i.e., 
  # places where there are more unique positive values than negative :
  if len(multval) > 0:
    # for each value and index in the array of multivals :
    for mv,i in zip(multval, where(g==True)[0]):
      zi = where(vs[:,i] == mv[0])[0]          # get rows with multiple posi's
      new = vs[zi]                             # copy them
      # for each value in the multival (other than the first, this was already
      # added in above loops), change the values in all rows of that column to 
      # the new value :
      for v in mv[1:]:
        new[:,i] = v
      # by stacking it inside this loop, we guarantee that if there are 
      # multiple multvals to process, we copy all of the new rows as well.
      vs = vstack((vs, new))                   # stack the result to the array

  print "\n\n final version space with any multivalues added :"
  print "--------------------------------------------------"
  print vs
  print "\n\n"
  
  return spc, gen, vs

def classify_cand_elim(test, train, classes):
  """
  Classify a set of test data <test> with corresponding training data <train> 
  and dictionary of two possible classes <classes>.  The positive index of 
  <classes> is 1, while the negative index is 0.
  """
  posi  = where(train[:,-1] == classes[1])[0]            # positive indices
  negi  = where(train[:,-1] == classes[0])[0]            # negative indices
  spc,  gen,  vs = candidate_elimination(train, classes) # cand. eliminate
  
  V_ce = []                                              # classified classes
  # iterate through each test instance and classify :
  for t in test:
    match = where((train[:,:-1] == t).all(axis=1))[0]    # where instances match
    # if any match, append the class of match :
    if len(match) > 0:
      V_ce.append(train[match[0]][-1])
    # else vote for the class :
    else:
      tally = []                               # final tally of votes
      # iterate through each element of the version space :
      for v in vs: 
        m = sum(v != '')                       # number of non-wild entries
        n = sum(v == t)                        # number of intersecting values
        # if non-wild entries match, vote positive : 
        if m == n:
          vote = classes[1]
        # otherwise, vote negative :
        else:
          vote = classes[0]
        tally.append(vote)                     # add the vote to the tally
      aye = sum(tally == classes[1])           # number of positive votes
      nay = sum(tally == classes[0])           # number of negative votes
      # aye wins :
      if aye > nay:
        V_ce.append(classes[1])                # append positive classification
      # nay wins :
      elif aye < nay:
        V_ce.append(classes[0])                # append negative classification
      # break ties randomly :
      else:
        V_ce.append(classes[randint(0,2)])     # append the random classif'n
  
  return array(V_ce)

def k_cross_validate(k, data, classify_ftn, classes): 
  """
  Perform <k> crossvalidation on data <data> with classification function
  <classify_ftn> and possible classes dictionary <classes>.
  """
  n   = shape(data)[0]         # number of samples
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

#===============================================================================
# formulate test dataset from the book :

data1 = array(['Sunny','Warm','Normal','Strong','Warm','Same','Enjoy Sport'], 
              dtype='|S12')
data2 = array(['Sunny','Warm','High','Strong','Warm','Same','Enjoy Sport'], 
              dtype='|S12')
data3 = array(['Rainy','Cold','High','Strong','Warm','Change','Do Not Enjoy'], 
              dtype='|S12')
data4 = array(['Sunny','Warm','High','Strong','Cool','Change','Enjoy Sport'], 
              dtype='|S12')
datat = vstack((data1, data2, data3, data4))


#===============================================================================
# calculate the version space for the dataset :

# truncate the data to only the unique instances (not required) :
#data = find_unique_row(data)

print "ENTIRE DATASET"
print "========================================================="
spc,  gen,  vs  = candidate_elimination(data, classes)

print "TEST DATASET FROM THE BOOK"
print "========================================================="
spct, gent, vst = candidate_elimination(datat, classes)

#===============================================================================
# perform classification with k-fold cross-validation :

k      = 10
result = k_cross_validate(k, data, classify_cand_elim, classes) 

print "percent correct: %.1f%%" % (100*average(result))



