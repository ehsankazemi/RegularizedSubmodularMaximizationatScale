import numpy as np
import sys
import math
import copy
from collections import defaultdict
from functions import *

def distorted_greedy(g, ell, dataset, k):
	#
	# This function implements the distorted greedy algorithm
	#
	auxiliary = set()
	current_val = 0
	S = []
	auxiliary = set()
	for i in range(k):
		max_margin = -1 * sys.maxsize
		best_element = 1
		best_auxiliary = set()
		for e in dataset['elements']:
			gain_g, new_auxiliary = g.marginal_gain(e, S, current_val, auxiliary)
			gain_ell = ell(dataset, e)
			# calculated the distorted gain of an element
			margin = math.pow(1 - 1.0 / k, k - (i+1)) * (gain_g) - gain_ell
			if margin > max_margin:
				max_margin = margin
				best_element = e
		if max_margin > 0:
			current_val, auxiliary = g.add_one_element(best_element, S, current_val, auxiliary)
			S.append(best_element)
	return S

def original_greedy(g, dataset, k):
	#
	# This is the greedy algorithm for a single function g.
	#
	auxiliary = set()
	current_val = 0
	S = []
	auxiliary = set()
	for i in range(k):
		max_margin = -1 * sys.maxsize
		best_element = 1
		best_auxiliary = set()
		for e in dataset['elements']:
			gain_g, new_auxiliary = g.marginal_gain(e, S, current_val, auxiliary)
			margin = gain_g 
			if margin > max_margin:
				max_margin = margin
				best_element = e
		if max_margin > 0:
			current_val, auxiliary = g.add_one_element(best_element, S, current_val, auxiliary)
			S.append(best_element)
	return S


def greedy(g, ell, dataset, k):
	#
	# This implements the greedy algorithm for g - ell
	#
	auxiliary = set()
	current_val = 0
	S = []
	auxiliary = set()
	for i in range(k):
		max_margin = -1 * sys.maxsize
		best_element = 1
		best_auxiliary = set()
		for e in dataset['elements']:
			gain_g, new_auxiliary = g.marginal_gain(e, S, current_val, auxiliary)
			gain_ell = ell(dataset, e)
			margin = gain_g - gain_ell
			if margin > max_margin:
				max_margin = margin
				best_element = e
				#best_auxiliary = copy.copy(new_auxiliary)
		if max_margin > 0:
			current_val, auxiliary = g.add_one_element(best_element, S, current_val, auxiliary)
			S.append(best_element)
	return S



def streaming_given_threshold(g, ell, r, tau, dataset, k):
	#
	# This function implements the THRESHOLD-STREAMING algorithm.
	#
	alpha = (2 * r + 1 + math.sqrt(4 * (r**2) + 1)) / 2.0
	S = []
	auxiliary = set()
	current_val = 0
	for e in dataset['elements']:
		gain_g, new_auxiliary = g.marginal_gain(e, S, current_val, auxiliary)
		gain_ell = ell(dataset, e)
		margin = gain_g - alpha * gain_ell
		if margin >= tau:
			current_val, auxiliary = g.add_one_element(e, S, current_val, auxiliary)
			S.append(e)
		if len(S) == k:
			return S
	return S


def streaming_threshold(g, ell, epsilon, dataset, k):
	#
	# This function implements the main proposed streaming algorithm in the paper:
	# DISTORTED-STREAMING
	#

	nb = 40 #number of different guesses for approximation ratios
	approx_ratios = [0.49 * (i + 1.0) / nb for i in range(nb)]

	#approximates value of beta
	betas = [((4 * x) / ((2 * x - 1)**2)) for x in approx_ratios]

	#approximations for r based on beta
	r_s = [b / (2 * math.sqrt(1 + 2 * b)) for b in betas]	
	
	#r_s = [0.05 * math.pow(1+epsilon, i) for i in range(50)]
	max_val = -1 * sys.maxsize
	max_set = []
	for i, r in enumerate(r_s):
		h = (2 * r + 1 - math.sqrt(4 * (r**2) + 1)) / 2.0
		#print(r, h)
		m = max([h * g.evaluate(e)[0] - r * ell(dataset, e) for e in dataset['elements']])
		opt = m 
		while opt < k * m:
			tau = opt / k
			S = streaming_given_threshold(g, ell, r, tau, dataset, k)
			opt *= (1 + epsilon)
			val = g.evaluate(S)[0] - ell(dataset, S) #changed
			if  val > max_val:
				max_val = val
				max_set = copy.copy(S)
		#print(i, betas[i], r, max_val, "------------")
	return max_set

def streaming_submodular_given_threshold(g, ell, tau, dataset, k):
	#
	# This function runs one instance of the sieve streaming algorithm
	# for a given threshold
	#
	S = []
	auxiliary = set()
	current_val = 0
	for e in dataset['elements']:
		gain_g, new_auxiliary = g.marginal_gain(e, S, current_val, auxiliary)
		gain_ell = ell(dataset, e)
		margin = gain_g - gain_ell
		if margin >= tau:
			current_val, auxiliary = g.add_one_element(e, S, current_val, auxiliary)
			S.append(e)
		if len(S) == k:
			return S
	return S

def streaming_submodular_threshold(g, ell, epsilon, dataset, k):
	#
	# This is the sieve streaming algorithm
	# Badanidiyuru, A., Mirzasoleiman, B., Karbasi, A., and Krause, A. 
	# Streaming Submodular Maximization:Massive Data Summarization on the Fly. 
	# In KDD, pp. 671–680, 2014.
	#
	#
	max_val = -1.0 * sys.maxsize
	max_set = []
	m = max([ g.evaluate(e)[0] - ell(dataset, e) for e in dataset['elements']])
	opt = m 
	while opt < k * m:
		tau = opt / (k * 1.0)
		S = streaming_submodular_given_threshold(g, ell, tau, dataset, k)
		opt *= (1 + epsilon)
		val = g.evaluate(S)[0] - ell(dataset, S)
		if  val > max_val:
			max_val = val
			max_set = copy.copy(S)
	return max_set

def vanilla_greedy_dynamic_program(g, dataset, k, costs):
	#
	# Implementation of the vanilla greedy dynamic program algorithm from the following paper:
	# Eyal Mizrachi, Roy Schwartz, Joachim Spoerhase, and Sumedha Uniyal. 
	# A Tight Approximation for Submodular Maximization with Mixed Packing and Covering Constraints. 
	# In International Colloquium on Automata, Languages, and Programming, (ICALP), pages 85:1–85:15, 2019.
	#
	p_max = 20
	costs = copy.copy(costs)

	for e in dataset['elements']:
		if int(costs[e] * p_max) == costs[e] * p_max:
			costs[e] = int(costs[e] * p_max)
		else:
			costs[e] = int(costs[e] * p_max) + 1


	T_list = defaultdict(list)
	T_set =  defaultdict(set)
	T_vals = dict()
	for q in range(k+1):
		for p in range(p_max+1):
			T_set[(q,p)] = set()
			T_list[(q,p)] = []
			T_vals[(q,p)] = 0

	for q in range(k+1):
		for p in range(p_max + 1):
			for e in dataset['elements']:
				if e not in T_set[(q,p)]:
					p_e = costs[e]
					p_prime = p_e + p
					if p_prime <= p_max and q < k:
						S = T_list[(q,p)] + [e]
						val_S = g.evaluate(S)[0]
						if val_S > T_vals[(q+1,p_prime)]:
							T_list[(q+1,p_prime)] = S
							T_set[(q+1,p_prime)] = set(S)
							T_vals[(q+1,p_prime)] = val_S
	best_val = -1 * sys.maxsize
	best_sol = []
	for q in range(k+1):
		for p in range(p_max + 1):
			if T_vals[(q,p)] > best_val:
				best_val = T_vals[(q,p)]
				best_sol = copy.copy(T_list[(q,p)])
	#print(best_val, best_sol)
	return(best_sol, best_val)



def FANTOM(f,dataset,constraint,epsilon = 0.2):
	#
	# Implementation of the FANTOM algorithm from:
	# Algorihm 3 from
	# Baharan Mirzasoleiman, Ashwinkumar Badanidiyuru, and Amin Karbasi. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# In ICML, pages 1358–1367, 2016.
	#
	n = len(dataset['elements'])
	vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
	M = max(vals_elements)
	U = []
	omgea = set([e for e in dataset['elements']])
	S = CandidateSol()
	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks
	gamma = (2 * p * M) / ( (p+1) * (2 * p + 1))
	R = []
	val = 1
	while  val <= n:
		R.append(val * gamma)                                                                                         
		val *= (1 + epsilon)

	for rho in R:
		omega = set([e for e in dataset['elements']])
		sols = iterated_GDT(f,dataset,constraint,omega,rho)
		for sol in sols:
			U.append(sol)

	final_soluton = []
	final_val = -1 * sys.maxsize
	for sol in U:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)

	#print(final_val, final_soluton)

	return final_val,  final_soluton

def iterated_GDT(f, dataset,constraint,ground,rho):
	#
	# sub-routine for FFANTOM
	# Algorihm 2: IGDT - Iterated greedy with density threshold
	# Mirzasoleiman, B., Badanidiyuru, A., and Karbasi, A. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# In ICML, pp. 1358–1367, 2016a.
	#
	S = CandidateSol()
	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks
	S_i = []
	for i in range(p+1):
		S = GDT(f,dataset,constraint,ground,rho)
		S_i.append(copy.copy(S.S_list))
		ground = ground - S.S_set
	return S_i

def GDT(f, dataset, constraint, ground, rho):
	#
	# Algorihm 1: GDT - Greedy with density threshold
	# Mirzasoleiman, B., Badanidiyuru, A., and Karbasi, A. 
	# Fast Constrained Submodular Maximization: Personalized Data Summarization. 
	# greedy with density threshold
	# runs the greedy on the on the groundser
	# an element is added if its marginal gain is larger than a threshold
	#
	S = CandidateSol()
	current_val = None
	flag = True
	while flag:
		flag = False
		cand = -1
		cand_val = -1 * sys.maxsize
		for e in ground:
			if constraint.is_add_feasible(e, S.S_list):
				val, __ = f.marginal_gain(e, S.S_list, current_val, S.auxiliary)
				if  val / (sum(constraint.costs[e]) + 0.000001) >= rho:
					if val > cand_val:
						cand = e
						cand_val = val
						flag = True
		if cand != -1:
			S.add_elements(cand)
			current_val, S.auxiliary = f.evaluate(S.S_list)
	return S

def fast_algorithms(f,dataset,constraint, epsilon = 0.2):
	#
	# Algorithm 10 from the following paper:
	# Badanidiyuru, A. and Vondrák, J. 
	# Fast algorithms for maximizing submodular functions. 
	# In ACM-SIAM symposium on Discrete algorithms (SODA), pp. 1497–1514, 2014.
	#
	n = len(dataset['elements'])
	vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
	M = max(vals_elements)

	p = constraint.nb_matroids
	ell = constraint.nb_knapsacks

	threshold_guesses = []
	guess = M / (p + ell * 1.0)
	while  guess <= (2 * constraint.max_cardinality * M) / (p + ell * 1.0):
		threshold_guesses.append(guess)
		guess *= (1 + epsilon)

	T_sols = []
	T_prime_sols = []
	for rho in threshold_guesses:
		vals_elements = [f.evaluate(e)[0] for e in dataset['elements']]
		rho_vals = [vals_elements[i] for i in range(n) if  (sum(constraint.costs[i]) == 0 or (vals_elements[i] / sum(constraint.costs[i])) >= rho )]
		if len(rho_vals) > 0:
			M_rho = max(rho_vals)
			#print(rho, M_rho, len(rho_vals))
			tau = M_rho
			S = CandidateSol()
			current_val = None
			S_rho = set()
			violate_flag = False
			while tau >= (epsilon / n ) * M_rho and constraint.check_costs(S.S_list):
				for j in dataset['elements']:
					if all([c <= 1 for c in constraint.costs[j]]):
						margin_j, __ = f.marginal_gain(j, S.S_list, current_val, S.auxiliary)
						if margin_j >= tau and (sum(constraint.costs[j]) == 0 or margin_j / sum(constraint.costs[j]) >= rho) and constraint.is_psystem_add_feasible(j, S.S_list):
							S.add_elements(j)
							current_val, S.auxiliary = f.evaluate(S.S_list)
							violate_flag = False
							if not constraint.check_costs(S.S_list):
								S_rho = copy.copy(S.S_set)
								T_rho = copy.copy(S.S_set)
								T_rho.remove(j)
								T_prime_pho = {j}
								violate_flag = True
								break
						if violate_flag:
							break
				if violate_flag:
					break
				tau = tau / (1.0 + epsilon)
		if 'violate_flag' in locals():
			if not violate_flag:
				T_rho = copy.copy(S.S_set)
				T_prime_pho = []
		if 'T_rho' in locals():
			T_sols.append(copy.copy(T_rho))
		if 'T_prime_pho' in locals():
			if len(T_prime_pho) > 0:
				T_prime_sols.append(copy.copy(T_prime_pho))

	final_soluton = []
	final_val = -1 * sys.maxsize
	for sol in T_prime_sols:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)

	for sol in T_sols:
		current_val, __ = f.evaluate(sol)
		if current_val > final_val:
			final_val = current_val
			final_soluton = copy.copy(sol)
	#print(final_val, final_soluton)
	return final_val, final_soluton