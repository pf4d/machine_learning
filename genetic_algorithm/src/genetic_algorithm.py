# genetic_algorithm.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the genetic algorithm to solve the 'knapsack' maximization
# problem.

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
# columns : Weight, Price

dirc   = '../data/'      # directory containing data
attrib = {0 : 'Weight', 
          1 : 'Price'}

# dataset :
f    = open(dirc + 'items.csv', 'r')
data = loadtxt(f, dtype='float', delimiter=",", skiprows=1)


#===============================================================================
# functions used :

def fitness(v, data, w_m):
  """
  calculate the fitness of a population matrix <v>.
  """
  w = data[0]                     # weight
  p = data[1]                     # price
                                  
  w_tot = dot(v, w)               # total weight
  p_tot = dot(v, p)               # total price
                                  
  p_tot[w_tot > w_m] = 0          # penalize overweight terms

  return p_tot

def crossover(v):
  """
  create children from population matrix <v>.
  """
  m,n = shape(v)
  shuffle(v)                      # shuffle the parents
  pairs = split(v, m/2)           # form parent pairs

  child = []
  for p in pairs:
    idx0 = randint(2, size=n)     # random binary array
    idx1 = abs(idx0 - 1)          # opposite array
    c    = p[0]*idx0 + p[1]*idx1  # child is combination of both
    child.append(c)
  
  return array(child)

def selection(v, data, w_m):
  """
  select the surviors of the population matrix <v>.
  """
  f = fitness(v, data, w_m)       # get the fitness of the population
  P = f / sum(f)                  # probability of selection
  c = cumsum(P)                   # cumulative probability

  idx = rand()                    # random index between 0 and 1
  sur = where(c > idx)[0][0]      # get the index of survivor
  return v[sur]

  

def genetic_algorithm(data, w_m):
  """
  """
  m,n  = shape(data)                # number of rows, columns
  p_s  = 100                        # population size
  pop  = randint(2, size=(p_s, m))  # generate initial population

  child = crossover(pop)            # create children
  pop   = vstack((pop, child))      # add the children to the population

  # find survivors :
  pop_n = []
  while len(pop_n) < p_s:
    survivor = selection(pop_n, data, w_m)
    pop_n.append(survivor)
  pop_n = array(pop_n)



#===============================================================================
# find solution :
w_m  = 200.0             # maximum weight





