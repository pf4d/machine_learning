# genetic_algorithm.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the genetic algorithm to solve the 'knapsack' maximization
# problem.

import csv
import sys

import matplotlib.gridspec as gridspec

from time         import time
from pylab        import *
from numpy.random import choice

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

def penalty(x, w_m):
  """
  penalty function for calculating fitness.
  """
  return 1/(x - w_m)

def fitness(V, data, w_m):
  """
  calculate the fitness of a population matrix <V>.
  """
  w = data[:,0]                   # weight
  p = data[:,1]                   # price
                                  
  w_tot = dot(V, w)               # total weight
  p_tot = dot(V, p)               # total price
  w_mp  = w_tot > w_m             # where overweight
 
  p        = ones(len(p_tot))            # penalty array
  p[w_mp]  = penalty(w_tot[w_mp], w_m)   # penalty term
  p_tot   *= p                           # penalize overweight terms

  return p_tot


def crossover(V):
  """
  create children from population matrix <V>.
  """
  m,n = shape(V)
  shuffle(V)                      # shuffle the parents
  pairs = split(V, m/2)           # form parent pairs

  child = []
  for p in pairs:
    idx0 = randint(2, size=n)     # random binary array
    idx1 = abs(idx0 - 1)          # opposite array
    c    = p[0]*idx0 + p[1]*idx1  # child is combination of both
    child.append(c)
  
  return array(child)


def selection(V, data, w_m):
  """
  select the surviors of the population matrix <V>.
  """
  f   = fitness(V, data, w_m)     # get the fitness of the population
  P   = f / sum(f)                # probability of selection
  c   = cumsum(P)                 # cumulative probability
  idx = rand()                    # random index between 0 and 1
  sur = where(c > idx)[0][0]      # get the index of survivor
  return sur


def genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma):
  """
  """
  m,n  = shape(data)                 # number of rows, columns
  pop  = randint(2, size=(p_s, m))   # generate initial population

  # perform algorithm for given number of generations:
  fit_avg = []
  fit_bst = []
  for i in range(gens):
    child    = crossover(pop)          # create children
    pop      = vstack((pop, child))    # add the children to the population
    L        = float(len(pop))         # size of population
    mut      = rand(L, m) < beta/L     # where to mutate
    pop[mut] = abs(pop[mut] - 1)       # mutate

    f     = fitness(pop, data, w_m)    # calculate the fitness
    best  = f.argsort()[-gamma:][::-1] # get indexes of gamma best
    pop_n = pop[best]                  # keep the gamma best individuals
    pop   = delete(pop, best, axis=0)  # remove the gamma best from pop.
    
    # find survivors :
    while len(pop_n) < p_s:
      idx    = arange(len(pop))                   # index array to choose from
      tourn  = choice(idx, alpha)                 # create the tournament
      sur    = selection(pop[tourn], data, w_m)   # find a survivor from tourn
      pop_n  = vstack((pop_n, pop[sur]))          # add survivor to new pop.
      pop    = vstack((pop[:sur], pop[sur+1:]))   # remove the sur from old pop.
    
    pop = pop_n                                   # kill off non-survivors
  
    fit_avg.append(average(f))
    fit_bst.append(best[0])
    print 'generation %i complete' % (i+1)

  return pop, fit_avg, fit_bst


 
#===============================================================================
# find solution :
w_m   = 200.0             # maximum weight
p_s   = 100               # population size
gens  = 1000              # number of generations
alpha = 20                # tournament size
beta  = 1.0               # mutation coefficient
gamma = 5                 # number of best individuals to keep


out = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

pop     = out[0]
fit_avg = out[1]
fit_bst = out[2]

plot(fit_avg, label='average')
#plot(fit_bst, label='best')
show()



