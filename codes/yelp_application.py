import os
import sys
import pickle
import numpy as np
from functions import *
import algorithms as algs
import time
import matplotlib.pyplot as plt
params = {'text.usetex': True,
              'figure.dpi': 400,
              'font.size': 12,
              'font.serif': [],
              'font.sans-serif': [],
              'font.monospace': [],
              'axes.labelsize': 14,
              'axes.titlesize': 12,
              'axes.linewidth': 1.0,
              'legend.fontsize': 12,
              'xtick.labelsize': 12,
              'ytick.labelsize': 12,
              'font.family': 'serif'}

lambdaa = 1.0

data_folder = os.path.join(os.getcwd(), 'dataset/yelp_data.csv')
dataset = get_yelp_data(data_folder, max_datapoints=1100, lambdaa=lambdaa)
#####################################################################################
location = 1 # sets the distance to the first point of interest in the dataset
dataset['dist_cost'] = dict()
for e in dataset['elements']:
	dataset['dist_cost'][e] = dataset['costs'][e][location]
dataset['orig_cost'] = dict()
for e in dataset['elements']:
	dataset['orig_cost'][e] = dataset['orig_distance'][e][location]
#####################################################################
g = FacilityLocation(dataset)
ell = distance_cost

dataset['reg_coef'] = 10 #normalizing coefficient of the cost
k=20 #cardinality constraint
epsilon = 0.2
####################
S = algs.original_greedy(g, dataset, k) #runs the greedy algorithm over the function g
print(g.evaluate(S)[0], ell(dataset,S)) #prints the quality of the result
####################


S = algs.streaming_submodular_threshold(g, ell, epsilon, dataset, k)
val = g.evaluate(S)[0] if len(S) != 0 else 0
ell_val = ell(dataset,S) if len(S) != 0 else 0
tot = 0
for e in S:
	tot+=dataset['orig_cost'][e]
avg =  (tot / len(S)) if len(S) != 0 else 0
print('sieve streaming:', avg, val - ell_val)


S = algs.distorted_greedy(g, ell, dataset, k)
val = g.evaluate(S)[0] if len(S) != 0 else 0
ell_val = ell(dataset,S) if len(S) != 0 else 0
tot = 0
for e in S:
	tot+=dataset['orig_cost'][e]
avg =  (tot / len(S)) if len(S) != 0 else 0
print('greedy:', avg, val - ell_val)


###### this part run the distorted streaming algorithm
g.oracle_calls = 0
epsilon = 0.5
S = algs.streaming_threshold(g, ell, epsilon, dataset, k)
val = g.evaluate(S)[0] if len(S) != 0 else 0
ell_val = ell(dataset,S) if len(S) != 0 else 0
tot = 0
for e in S:
	tot+=dataset['orig_cost'][e]
avg =  (tot / len(S)) if len(S) != 0 else 0
print('distorted streaming:', avg, val - ell_val)




#runs the algorithm for different knapsack costs
fantom = []
fast = []
greedy_dp = []
for regulizer in [0.01, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20., 100.0]:
	costs = defaultdict(list)
	for e in dataset['elements']:
		costs[e] = [dataset['dist_cost'][e] * regulizer]
	constraint = KnapsackConstraint(dataset, costs, k)

	g.oracle_calls = 0
	val, S = algs.FANTOM(g, dataset, constraint)
	val = g.evaluate(S)[0] if len(S) != 0 else 0
	ell_val = ell(dataset,S) if len(S) != 0 else 0
	tot = 0
	for e in S:
		tot+=dataset['orig_cost'][e]
	avg =  (tot / len(S)) if len(S) != 0 else 0
	print(regulizer, val - ell_val)

	g.oracle_calls = 0
	val, S = algs.fast_algorithms(g, dataset, constraint)
	val = g.evaluate(S)[0] if len(S) != 0 else 0
	ell_val = ell(dataset,S) if len(S) != 0 else 0
	tot = 0
	for e in S:
		tot+=dataset['orig_cost'][e]
	avg =  (tot / len(S)) if len(S) != 0 else 0
	fast.append([val, ell_val, g.oracle_calls, avg])


	costs = copy.copy(dataset['costs'])
	for e in dataset['elements']:
		costs[e] = dataset['dist_cost'][e] * regulizer
	g.oracle_calls = 0
	S, val = algs.vanilla_greedy_dynamic_program(g, dataset, k, costs)
	val = g.evaluate(S)[0] if len(S) != 0 else 0
	ell_val = ell(dataset,S) if len(S) != 0 else 0
	tot = 0
	for e in S:
		tot+=dataset['orig_cost'][e]
	avg =  (tot / len(S)) if len(S) != 0 else 0
	print(regulizer, val - ell_val)
	greedy_dp.append([val, ell_val, g.oracle_calls, avg])
