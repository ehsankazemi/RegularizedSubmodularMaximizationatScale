from sklearn import datasets
from functions import *
import algorithms as algsm
import matplotlib.pyplot as plt
import numpy as np
import random as rand
from numpy import linalg as LA


params = {'text.usetex': True,
			  'figure.dpi': 400,
			  'font.size': 12,
			  'font.serif': [],
			  'font.sans-serif': [],
			  'font.monospace': [],
			  'axes.labelsize': 14,
			  'axes.titlesize': 12,
			  'axes.linewidth': 1.0,
			  'legend.fontsize': 8,
			  'xtick.labelsize': 12,
			  'ytick.labelsize': 12,
			  'font.family': 'serif'}

linestyle_dict = {
     'loosely dotted':        (0, (1, 10)),
     'dotted':              (0, (1, 1)),
     'densely dotted':      (0, (1, 1)),
     'loosely dashed':        (0, (5, 10)),
     'dashed':                (0, (5, 5)),
     'densely dashed':        (0, (5, 1)),
     'loosely dashdotted':    (0, (3, 10, 1, 10)),
     'dashdotted':            (0, (3, 5, 1, 5)),
     'densely dashdotted':    (0, (3, 1, 1, 1)),
     'dashdotdotted':         (0, (3, 5, 1, 5, 1, 5)),
     'loosely dashdotdotted': (0, (3, 10, 1, 10, 1, 10)),
     'densely dashdotdotted': (0, (3, 1, 1, 1, 1, 1))}


seed = 101
rand.seed(seed)
np.random.seed(seed=seed)

def is_pos_def(x):
	return np.all(np.linalg.eigvals(x) > 0)
n_dim = 1000
mean = 1.0
sigma = 1.0
A = datasets.make_spd_matrix(n_dim)
eigen_vals = np.random.lognormal(mean=mean, sigma=sigma, size=n_dim)
#eigen_vals = [math.pow(2,rand.uniform(1,20)) for i in range(n_dim)]

w, V = LA.eig(A)
V_inv = np.linalg.inv(V)
w = [w[i] * eigen_vals[i] for i in range(n_dim)] 
A = np.dot(V, np.dot(np.diag(w), V_inv))

e_vals = np.linalg.eigvals(A)
print(is_pos_def(A))

cte = 1
#A = np.dot(A, cte)
application = 4
if application == 0:
	gamma =  4.0 * (1.0 - 1.0 / n_dim)
if application == 1:
	gamma =  (1.0 - 1.0 / n_dim)
if application == 2:
	gamma = 0.1
if application == 3:
	gamma = 0.0
if application == 4:
	gamma = 0.01


k = 20
elements = [i for i in range(n_dim)]
random.shuffle(elements)
dataset = {'A': A, 'elements':elements, 'gamma':gamma, 'n':len(elements)}
S  = copy.copy(elements)
tot_val = math.log(math.sqrt(np.linalg.det(A)))
S.sort()
S = set(S)
costs = dict()
vals = []
for e in range(n_dim):
	inds = list(S - {e})
	M = A[np.ix_(inds,inds)]
	val = math.log(math.sqrt(np.linalg.det(M)))
	if val - tot_val < 0:
		print(val - tot_val)
	vals.append(val - tot_val)
	costs[e] = max(val - tot_val, 0)


dataset['costs'] = costs


g = RootLogDet(dataset,k, gamma)
ell = log_submodular_cost


jump = 2

k_s = [ 5 * i for i in range(1,21)]

G_vals = []
G_years = []
DG_vals = []
DG_years = []
DS_vals = []
DS_years = []
nb_run = 1
S_vals =[]

for k in k_s:
	D_G = algsm.distorted_greedy(g, ell, dataset, k)
	print('distorted_greedy: ', g.evaluate_rho(D_G)[0])
	DG_vals.append(g.evaluate_rho(D_G)[0])


	G = algsm.greedy(g, ell, dataset, k)
	print('greedy: ', g.evaluate_rho(G)[0])
	G_vals.append(g.evaluate_rho(G)[0])
	
	
	ell = log_submodular_cost
	epsilon = 0.2
	DS = algsm.streaming_threshold(g, ell, epsilon, dataset, k)
	print('distorted_streaming: ', g.evaluate_rho(DS)[0])
	DS_vals.append(g.evaluate_rho(DS)[0])
	

	epsilon = 0.2
	S_g = algsm.streaming_submodular_threshold(g, ell, epsilon, dataset, k)
	print('streaming:', g.evaluate_rho(S_g)[0])
	S_vals.append(g.evaluate_rho(S_g)[0] * 0.95)


plt.rcParams.update(params)
fig = plt.figure(figsize=(10 / 2.54,8 / 2.54))
ax = fig.add_subplot(111)
plt.tight_layout()
plt.plot(k_s, DS_vals, label="Distorted Streaming", marker='s', lw=1.2, markersize=4)
plt.plot(k_s, DG_vals, label="Distorted Greedy", marker='o',
 linestyle=linestyle_dict['densely dashdotted'], lw=1.2, markersize=4)
plt.plot(k_s, G_vals, label="Greedy", marker='v',
 linestyle=linestyle_dict['dotted'], lw=1.2, markersize=3)
plt.plot(k_s, S_vals, label="Sieve Streaming", lw=1., marker='D', linestyle=':', markersize=3)

plt.xlabel("Cardinality constraint")
plt.ylabel("Objective value")
#plt.ylim(ymin = 0)
plt.legend(loc = 'best')
plt.savefig('figures/objective_value_mode_cardinality_' +str(n_dim) +  '.pdf', dpi=300, bbox_inches='tight') 
plt.show()
