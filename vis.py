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
from scipy.ndimage.morphology import binary_dilation
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
	G.add_nodes_from(np.arange(r))

	for i in range(d):
		if omega[edge_idx,i]!=0:
			G.add_edge(idx[i][0],idx[i][1])
	## Delete nodes with degree 0
	G.remove_nodes_from(list(nx.isolates(G)))
	print('connected components number:', nx.number_connected_components(G))
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
	task = 'WM'
	omega = np.load('0.0009_'+task+'.npy') #p*2p
	omega = omega[:,omega.shape[0]:]
	print(omega.shape)
	print(np.count_nonzero(omega))

	# visulize omega_fs nz entries
	plt.figure()
	omega_vis = omega.copy()
	omega_vis[omega_vis!=0]=1

	for _ in range(3):
		omega_vis = binary_dilation(omega_vis)
	 
	plt.imshow(omega_vis)
	plt.axis('off')
	plt.savefig('fs_'+task+'.png', bbox_inches = 'tight', pad_inches = 0)
	plt.show()

	idx_dict = build_dict(len(r_name))
	# relative info
	# 1: which structral edges are the top important k? [per col]
	s_topk = 10
	omega_vis = omega.copy()
	omega_vis[omega_vis!=0]=1
	s_sum = np.sum(omega_vis,axis=0)
	print(s_sum.max(),s_sum.mean(),s_sum.min())
	sorted_idx_s = np.argsort(s_sum)[::-1]
	for i in range(s_topk):
		print('\ncorrelated function number:',s_sum[sorted_idx_s[i]])
		idx = idx_dict[sorted_idx_s[i]]
		print(r_name[idx[0]],r_name[idx[1]])

	# 2: does these edges exists for every subject in the original structral data?
	vec, _ = data_prep(upenn=False)
	vec_s = vec.copy()
	# vec_s[vec_s!=0]=1
	tmp = np.sum(vec_s,axis=0)
	print('\nmax strength of an edge:', tmp.max())
	print('top important k edges strength:',tmp[sorted_idx_s[:s_topk]],'\n') #emm seems the answer is no

	# 3: for each functional activation, plot "which structral edges is correlated with it" [per row]
	f_topk = 3
	f_sum = np.sum(omega_vis,axis=1)
	print(f_sum.max(),f_sum.mean(),f_sum.min())
	sorted_idx_f = np.argsort(f_sum)[::-1]
	for i in range(f_topk):
		idx = idx_dict[sorted_idx_f[i]]
		print('\ncorrelated structral edge number:',f_sum[sorted_idx_f[i]])
		print(r_name[idx[0]],r_name[idx[1]])
		plot_edge(omega,r_name,idx_dict,sorted_idx_f[i])
