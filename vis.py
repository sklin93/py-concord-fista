"""Drawing the overlapped network across different functions"""
import numpy as np
from scipy.io import loadmat

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from hcp_cc import data_prep

import yaml
with open('config.yaml') as info:
    info_dict = yaml.load(info)

def build_dict(r):
	idx = {}
	ctr = 0
	for i in range(r-1):
		for j in range(i+1,r):
			idx[ctr] = [i,j]
			ctr = ctr + 1
	return idx

def plot_edge(omega, r_name, idx, edge_idx):
	[d,_] = omega.shape
	r = len(r_name) # num regions

	G = nx.Graph()
	G.add_nodes_from(np.arange(len(r)))

	ctr = 0
	for i in range(d-1):
		for j in range(i+1,d):
			if omega[i,j] != 0:
				G.add_edge(idx[i][0],idx[i][1])
				G.add_edge(idx[j][0],idx[j][1])
				ctr = ctr+1
	print(ctr)
	## Delete nodes with degree 0
	to_del = nx.isolates(G)
	G.remove_nodes_from(to_del)

	## Draw
	options = {'node_color': '#FA8072', 'edge_color': '#2C3E50', 
				'node_size': 400,'width': 0.8,}
	labels = {}
	for i in range(r):
		labels[i]=r_name[i]
	G = nx.relabel_nodes(G,labels)
	nx.draw(G, with_labels=True, **options) #font_weight='bold'
	plt.show()


if __name__ == '__main__':
	r_name = info_dict['data']['aal']

	omega = np.load('0.1.npy') #p*2p
	omega = omega[:,omega.shape[0]:]
	print(omega.shape)
	print(np.count_nonzero(omega))

	# visulize omega_fs nz entries
	plt.figure()
	omega_vis = omega.copy()
	omega_vis[omega_vis!=0]=1
	plt.imshow(omega_vis)
	plt.show()

	idx_dict = build_dict(len(r_name))
	# relative info
	# 1: which structral edges are the top important k? [per col]
	s_topk = 5
	sorted_idx = np.argsort(np.sum(omega,axis=0))[::-1]
	for i in range(s_topk):
		print(sorted_idx[i])
		idx = idx_dict[sorted_idx[i]]
		print(idx)
		print(r_name[idx[0]],r_name[idx[1]])

	# 2: does these edges exists for every subject in the original structral data?
	vec, _ = data_prep(upenn=True)
	import ipdb; ipdb.set_trace()
	vec_s = vec.copy()
	vec_s[vec_s!=0]=1
	tmp = np.sum(vec_s,axis=0)
	print(tmp[sorted_idx[:s_topk]]) #emm seems the answer is no
	
	# 3: for each functional activation, plot "which structral edges is correlated with it" [per row]
	