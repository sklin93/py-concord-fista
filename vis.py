"""Drawing the overlapped network across different functions"""
import numpy as np
import sys
from scipy.io import loadmat

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import networkx as nx

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from hcp_cc import data_prep
from common_nonzero import load_omega, nz_share
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

def get_cmap(n, name='hsv'):
	'''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct
	RGB color; the keyword argument name must be a standard mpl colormap name.'''
	return plt.cm.get_cmap(name, n)
	
def get_graph(omega, r, idx, edge_idx):
	[d,_] = omega.shape

	G = nx.Graph()
	G.add_nodes_from(np.arange(r))

	for i in range(d):
		if omega[edge_idx,i]!=0:
			G.add_edge(idx[i][0], idx[i][1])
	## Delete nodes with degree 0
	G.remove_nodes_from(list(nx.isolates(G)))
	return G

def plot_edge(omega, r_name, idx, edge_idx):
	r = len(r_name)
	G = get_graph(omega, r, idx, edge_idx)
	print('connected components number:', nx.number_connected_components(G))
	## Draw
	options = {'node_color': '#FA8072', 'edge_color': '#2C3E50', 
				'node_size': 400,'width': 0.8,}
	labels = {}
	for i in range(r):
		labels[i]=r_name[i]
	G = nx.relabel_nodes(G, labels)
	nx.draw(G, with_labels=True, **options) #font_weight='bold'
	plt.show()
	nx.draw(nx.k_core(G, 2), with_labels=True, **options)
	plt.show()

if __name__ == '__main__':
	fdir = 'fs_results/'
	task = sys.argv[1]
	lam = float(sys.argv[2])

	if task == 'resting':
		r_name = info_dict['data']['aal']
	else:
		r_name = info_dict['data']['hcp']
	# load omega
	if task == 'resting' or task[:6] == 'syn_sf':
		# omega = load_omega(task, mid='_train_', lam=9e-05)
		B = np.load(fdir+str(lam)+'_mrce_B_'+task+'.npy')
		omega = B
	else:
		# omega = load_omega(task, mid='_er_train_hcp2_', lam=0.0014)
		B = np.load(fdir+str(lam)+'_mrce_B_'+task+'.npy')
		omega = B # just for visualization purpose (tmp, #TODO cleanup)
	# omega = np.load('fs_results/direct_9e-05.npy')[:3403, 3403:]
	# omega = np.load('fs_results/9e-05_train_syn_sf_sf.npy')[:, 3403:]

	# visulize omega_fs nz entries
	plt.figure()
	omega_vis = omega.copy()
	omega_vis[omega_vis!=0]=1

	for _ in range(3):
		omega_vis = binary_dilation(omega_vis)
	 
	# plt.imshow(omega_vis, cmap='Blues')
	plt.imshow(omega_vis)
	plt.axis('off')
	plt.savefig(fdir+'fs_'+task+'.png', bbox_inches = 'tight', pad_inches = 0)
	plt.show()

	# if using synthetic data, visualize ground truth
	if task == 'syn_sf':
		import pickle
		with open('data-utility/syn_sf_sf.pkl', 'rb') as f:
			gt_w = pickle.load(f)['W']
			gt_w[gt_w!=0] = 1
			for _ in range(3):
				gt_w = binary_dilation(gt_w)
			plt.imshow(gt_w.T, cmap='Blues')
			plt.axis('off')
			plt.show()

	idx_dict = build_dict(len(r_name))
	# relative info
	# 1: which structral edges are the top important k? [per col]
	s_topk = 10
	omega_vis = omega.copy()
	omega_vis[omega_vis!=0]=1
	s_sum = np.sum(omega_vis, axis=0)

	'''
	# plot nz entry number distribution
	import seaborn as sns
	sns.distplot(s_sum, kde=False)
	plt.show()
	import powerlaw
	plt.figure()
	pl_fit = powerlaw.Fit(s_sum)
	pl_R, pl_p = pl_fit.distribution_compare('power_law', 'lognormal') #more log_normal
	fig = pl_fit.plot_cdf(linewidth=3, color='b')
	pl_fit.power_law.plot_cdf(ax=fig, color='g', linestyle='--')
	pl_fit.lognormal.plot_cdf(ax=fig, color='r', linestyle='--')
	plt.show()
	'''
	import ipdb; ipdb.set_trace()

	print(s_sum.max(), s_sum.mean(), s_sum.min())
	sorted_idx_s = np.argsort(s_sum)[::-1]
	# for i in range(s_topk):
	for i in range(np.count_nonzero(s_sum)):
		print('\ncorrelated function number:', s_sum[sorted_idx_s[i]])
		idx = idx_dict[sorted_idx_s[i]]
		print(r_name[idx[0]], r_name[idx[1]])

	# 2: does these edges exists for every subject in the original structral data?
	vec, _ = data_prep(task)
	vec_s = vec.copy()
	# vec_s[vec_s!=0]=1
	tmp = np.sum(vec_s,axis=0)
	print('\nmax strength of an edge:', tmp.max())
	print('top important k edges strength:', tmp[sorted_idx_s[:s_topk]], '\n') #emm seems the answer is no
	import ipdb; ipdb.set_trace()
	# 3: for each functional activation, plot "which structral edges is correlated with it" [per row]
	f_topk = 3
	f_sum = np.sum(omega_vis, axis=1)
	print(f_sum.max(),f_sum.mean(), f_sum.min())
	sorted_idx_f = np.argsort(f_sum)[::-1]
	for i in range(f_topk):
		idx = idx_dict[sorted_idx_f[i]]
		print('\ncorrelated structral edge number:', f_sum[sorted_idx_f[i]])
		print(sorted_idx_f[i], r_name[idx[0]], r_name[idx[1]])
		plot_edge(omega, r_name, idx_dict, sorted_idx_f[i])
	import ipdb; ipdb.set_trace()
	# 4: check connected component for all function edges
	cc = 0
	ctr = 0
	for i in range(omega.shape[0]):
		G = get_graph(omega, len(r_name), idx_dict, i)
		if f_sum[i] > 1:
			print('nz:', f_sum[i], 'cc:', nx.number_connected_components(G))
			cc += nx.number_connected_components(G)
			ctr += 1
	print('Average connected component number:', cc/ctr)