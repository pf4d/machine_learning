# genetic_algorithm.py
# Evan Cummings
# CSCI 544 - Machine Learning
# Spring 2014 - Doug Raiford

# This script uses the genetic algorithm to solve the 'knapsack' maximization
# problem.
#
# mutations    : time to compute: 32.55 seconds
#                time to compute: 32.57 seconds
# no mutations : time to compute: 32.10 seconds
#                time to compute: 32.23 seconds


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

def greedy(data, w_m):
  """
  calculate the best fit utilizing the greedy approach (add best price density 
  until the bag is full), with weight/price matrix <data> and max weight <w_m>.
  """
  w   = data[:,0]                 # weight
  p   = data[:,1]                 # price
  rho = p/w                       # density
  idx = argsort(rho)[::-1]        # sorted density indexes
  w_a = cumsum(w[idx])            # weight sums
  p_a = cumsum(p[idx])            # price sums
  tru = w_a <= w_m                # where the weight is below threshold
  w_a = w_a[tru]                  # truncate weight total
  p_a = p_a[tru]                  # truncate price total
  return w_a[-1], p_a[-1]         # return the best 


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
  Genetic algorithm solves the knapsack maximization problem for data matrix 
  <data>, with maximum weight <w_m>, population size <p_s>, number of
  generations <gens>, tournament selection size <alpha>, mutation rate 
  coefficient <beta>, and elitism parameter <gamma> (the <gamma> best 
  individuals are saved at every generation).

  Returns a tuple containing the final population, array of average fitnesses, 
  best fitnesses, and best fit weights.
  """
  m,n  = shape(data)                 # number of rows, columns
  pop  = randint(2, size=(p_s, m))   # generate initial population

  # perform algorithm for given number of generations:
  fit_avg   = []
  fit_bst   = []
  fit_bst_w = []
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
      pop_n  = vstack((pop_n, pop[tourn][sur]))   # add survivor to new pop.
      pop    = vstack((pop[:sur], pop[sur+1:]))   # remove the sur from old pop.
    
    pop = pop_n                                   # kill off non-survivors
    
    f     = fitness(pop, data, w_m)    # calculate the fitness
    best  = f.argsort()[-1]            # get index to best individual
    fit_avg.append(average(f))
    fit_bst.append(f[best])
    fit_bst_w.append(dot(pop[best], data[:,0]))
    print 'generation %i complete' % (i+1)

  return pop, fit_avg, fit_bst, fit_bst_w

 
#===============================================================================
# greedy approach :
w_g, p_g = greedy(data, 200) 

print 'greedy solution (best density): weight = %i, price = %i' % (w_g, p_g)


#===============================================================================
# find solution :
w_m   = 200.0       # maximum weight
p_s   = 100         # population size
gens  = 500         # number of generations
gamma = 5           # number of best individuals to keep

alpha = 80          # tournament size
beta  = 1.0         # mutation coefficient
out1  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

alpha = 1           # tournament size
beta  = 1.0         # mutation coefficient
out2  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

alpha = 2           # tournament size
beta  = 2.0         # mutation coefficient
out3  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

alpha = 10          # tournament size
beta  = 1.0         # mutation coefficient
out4  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

alpha = 30          # tournament size
beta  = 1.0         # mutation coefficient
out5  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)

alpha = 30          # tournament size
beta  = 3.0         # mutation coefficient
out6  = genetic_algorithm(data, w_m, p_s, gens, alpha, beta, gamma)


#===============================================================================
# plot the solutions :
out   = [out1,out2,out3,out4,out5,out6]
names = [r'$\alpha = 80$, $\beta = 1$',
         r'$\alpha = 1$, $\beta = 1$',
         r'$\alpha = 2$, $\beta = 2$',
         r'$\alpha = 10$, $\beta = 1$',
         r'$\alpha = 30$, $\beta = 1$',
         r'$\alpha = 30$, $\beta = 3$']

fig = figure(figsize=(14,8))

for i,(o,n) in enumerate(zip(out,names)):
  idx = 231 + i
  ax  = fig.add_subplot(idx)
  pop       = o[0]
  fit_avg   = o[1]
  fit_bst   = o[2]
  fit_bst_w = o[3]
 
  max_g = argmax(fit_bst)           # generation of best individual
  max_p = fit_bst[max_g]            # total value of best individual
  max_w = fit_bst_w[max_g]          # total weight of best individual
 
  ax.text(10,  1100, 'gen : %i'   % max_g, color = 'r')
  ax.text(150, 1100, 'max w : %i' % max_w, color = 'r')
  ax.text(340, 1100, 'max p : %i' % max_p, color = 'r')
  
  ax.plot(fit_avg, lw=2.0, label='avg')
  ax.plot(fit_bst, lw=2.0, label='best')
  if i == 0 or i == 3:
    ax.set_ylabel('fitness')
  if i == 3 or i == 4 or i == 5:
    ax.set_xlabel('generation')
  ax.set_title(n)
  ax.set_ylim([0,1200])
  leg = ax.legend(loc='lower right')
  leg.get_frame().set_alpha(0.5)
  ax.grid()

tight_layout()
savefig('../doc/images/out.png', dpi=300)
show()



