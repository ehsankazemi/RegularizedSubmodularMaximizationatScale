from functions import *
import algorithms as algsm
import matplotlib.pyplot as plt
import os

#this file compare the performance of different algorithm on the graph application

ind = 3
files = ['wiki-Vote', '1912', 'email-Eu-core', '0']
file_name = files[ind]
#file_name = '1912'
#data_file = 'dataset/HR_edges.csv'
#data_file = 'dataset/0.edges'

data_file = 'dataset/' + file_name +'.txt'


insert_flag = True



if file_name == 'wiki-Vote':
	nb_k  = 20
	jump = 100

if file_name == '1912':
	nb_k  = 20
	jump = 10

if file_name == 'email-Eu-core':
	nb_k  = 20
	jump = 15

if file_name == '0':
	nb_k  = 20
	jump = 10

nb_run = 10 #averages the results over this number of experiments

G_vals = []
DG_vals = []
DS_vals = []
Sl_vals = []
Gl_vals = []
DG_vals = []
DSl_vals = []
Sl_vals = []


G_vals = []
S_vals = []
DS_val = []

G_l = []
S_l = []
DS_l = []

G_o = []
S_o = []
DS_0 = []


dataset = get_graph_data(data_file, directed = True)
print(len(dataset['elements']))
g = VertexCover(dataset)
ell = node_cost


k_s = [(i+1) * jump for i in range(nb_k)]
if insert_flag:
	k_s.insert(0, 10)
k_s = list(set(k_s))
k_s.sort()



result_file_name = 'result_'+ files[ind] + '.txt' #stores the result at this file

fto = open(result_file_name, "w")
fto.close()

for k in k_s:

	result = []
	result.append(k)


	g.oracle_calls = 0
	D_G = algsm.distorted_greedy(g, ell, dataset, k)
	print('distorted_greedy: ', g.evaluate(D_G)[0] - ell(dataset, D_G))
	DG_vals.append(g.evaluate(D_G)[0] - ell(dataset, D_G))

	result.append(g.evaluate(D_G)[0])
	result.append(ell(dataset, D_G)) 
	result.append(g.oracle_calls)

	g_val = 0
	ds_val = 0
	s_val = 0

	g_l = 0
	ds_l = 0
	s_l = 0

	g_o = 0
	ds_o = 0
	s_o = 0
	
	for r in range(nb_run):
		print(r, k , " --------------")
		random.shuffle(dataset['elements'])
		g = VertexCover(dataset)
		g.oracle_calls = 0
		G = algsm.greedy(g, ell, dataset, k)
		print('greedy: ', g.evaluate(G)[0] - ell(dataset, G))
		g_val += g.evaluate(G)[0]
		g_l += ell(dataset, G)
		g_o += g.oracle_calls

		
		ell = node_cost
		epsilon = 0.2
		g.oracle_calls = 0
		DS = algsm.streaming_threshold(g, node_cost, epsilon, dataset, k)
		print('distorted_streaming: ', g.evaluate(DS)[0] - ell(dataset, DS))
		ds_val += g.evaluate(DS)[0]
		ds_l += ell(dataset, DS)
		ds_o += g.oracle_calls

		

		epsilon = 0.2
		g.oracle_calls = 0
		S_g = algsm.streaming_submodular_threshold(g, node_cost, epsilon, dataset, k)
		print(' streaming: ', g.evaluate(S_g)[0] - ell(dataset, S_g))
		s_val += g.evaluate(S_g)[0]
		s_l += ell(dataset, S_g)
		s_o += g.oracle_calls
		
	result.append(g_val / (nb_run * 1.0))
	result.append(g_l / (nb_run * 1.0))
	result.append(g_o / (nb_run * 1.0))

	result.append(ds_val / (nb_run * 1.0))
	result.append(ds_l / (nb_run * 1.0))
	result.append(ds_o / (nb_run * 1.0))

	result.append(s_val / (nb_run * 1.0))
	result.append(s_l / (nb_run * 1.0))
	result.append(s_o / (nb_run * 1.0))
	print()
	print()
	print(result)
	fto = open(result_file_name, "a")
	for val in result:
		fto.write(str(val) + "\t")
	fto.write("\n")	


#load parameters for plotting
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



#plots the results

rows =[]
with open(result_file_name, 'r') as f:
	for line in f:
		parsed = [float(val) for val in line.replace('\t\n','').split()]
		rows.append(parsed)
reuslts = []
for col in zip(*rows):
	reuslts.append(list(col))
#print(reuslts[1])


dg = [rows[i][1] - rows[i][2] for i in range(len(rows))]
g = [rows[i][4] - rows[i][5] for i in range(len(rows))]
ds = [rows[i][7] - rows[i][8] for i in range(len(rows))]
s = [rows[i][10] - rows[i][11] for i in range(len(rows))]

plt.rcParams.update(params)
fig = plt.figure(figsize=(10 / 2.54,8 / 2.54))
ax = fig.add_subplot(111)
plt.tight_layout()
plt.plot(reuslts[0], ds, label="Distorted Streaming", marker='s', lw=1.2, markersize=4)
plt.plot(reuslts[0], dg, label="Distorted Greedy", marker='o',
 linestyle=linestyle_dict['densely dashdotted'], lw=1.2, markersize=4)
plt.plot(reuslts[0], g, label="Greedy", marker='v',
 linestyle=linestyle_dict['dotted'], lw=1.2, markersize=3)
plt.plot(reuslts[0], s, label="Sieve Streaming", lw=1., marker='D', linestyle=':', markersize=3)

plt.xlabel("Cardinality constraint")
plt.ylabel("Objective value")
plt.ylim(ymin = 0)
plt.legend(loc = 'best')
plt.savefig('objective_value_' + file_name + '.pdf', dpi=300, bbox_inches='tight') 

plt.show()



