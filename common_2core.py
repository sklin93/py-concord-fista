import numpy as np 
import networkx as nx
import sys
from hcp_cc import data_prep
from vis import build_dict, get_graph
from common_nonzero import load_omega
import yaml

def common_nodes(omega, r_name, idx_dict, save=True):
	r = len(r_name)
	appearance_nodes = np.zeros(r)
	for i in range(omega.shape[0]):
		G = get_graph(omega, r, idx_dict, i)
		subG = nx.k_core(G,2)
		cur_nodes = list(subG.nodes)
		appearance_nodes[cur_nodes] += 1

	size = appearance_nodes/appearance_nodes.max()
	if save: np.save('size.npy',size)

	sorted_idx = np.argsort(appearance_nodes)[::-1]
	for i in range(r):
		cur_idx = sorted_idx[i]
		if appearance_nodes[cur_idx]!=0:
			print(appearance_nodes[cur_idx], r_name[cur_idx])
	return appearance_nodes

def common_edges(omega, r_name, idx_dict, vis_dir='', save=True):
	r = len(r_name)
	appearance_edges = np.zeros((r,r))
	for i in range(omega.shape[0]):
		G = get_graph(omega, r, idx_dict, i)
		subG = nx.k_core(G,2)
		for j in range(len(subG.edges)):
			appearance_edges[list(subG.edges)[j]] += 1
	appearance_edges += appearance_edges.T

	flat_app_edge = []
	for i in range(r-1):
		for j in range(i+1,r):
			flat_app_edge.append(appearance_edges[i,j])
	sorted_idx = np.argsort(flat_app_edge)[::-1]
	print(len(flat_app_edge))
	for i in range(len(flat_app_edge)):
		cur_idx = sorted_idx[i]
		if flat_app_edge[cur_idx]!=0:
			print(flat_app_edge[cur_idx], r_name[idx_dict[cur_idx][0]], r_name[idx_dict[cur_idx][1]])
	if save: np.savetxt(vis_dir+'common_edge.edge', appearance_edges, fmt='%i')
	return flat_app_edge

# common triangles
def nC3_dict(r):
	"""n choose 3 mapping"""
	idx = {}
	ctr = 0
	for i in range(r-2):
		for j in range(i+1,r-1):
			for k in range(j+1,r):
				idx[ctr] = [i,j,k]
				ctr+=1
	inv_idx = {tuple(v): k for k, v in idx.items()}
	return idx, inv_idx

def common_triangles(omega, r_name, idx_dict):
	r = len(r_name)
	idx_dict_3, inv_idx_dict_3 = nC3_dict(r)
	appearance_triangles = np.zeros(len(idx_dict_3))
	for f_idx in range(omega.shape[0]):
		G = get_graph(omega, r, idx_dict, f_idx)
		subG = nx.k_core(G,2)
		if len(subG.edges) > 2:
			tmp_list = list(subG.edges)
			while len(tmp_list)>0:
				node1 = tmp_list[0][0]
				node2 = tmp_list[0][1]
				to_delete = [tmp_list[0]]
				for i in range(1,len(tmp_list)):
					if node1 in tmp_list[i]:
						node3 = tmp_list[i][0] if node1==tmp_list[i][1] else tmp_list[i][1]
						if ((node2,node3) in tmp_list) or ((node3,node2) in tmp_list):
							appearance_triangles[inv_idx_dict_3[tuple(sorted([node1,node2,node3]))]]+=1
							to_delete.append(tmp_list[i])
							if (node2,node3) in tmp_list:
								to_delete.append((node2,node3))
							else:
								to_delete.append((node3,node2))
				for item in to_delete: tmp_list.remove(item)
	sorted_idx = np.argsort(appearance_triangles)[::-1]
	for i in range(len(appearance_triangles)):
		cur_idx = sorted_idx[i]
		if appearance_triangles[cur_idx]!=0:
			print(appearance_triangles[cur_idx],r_name[idx_dict_3[cur_idx][0]],
					r_name[idx_dict_3[cur_idx][1]],r_name[idx_dict_3[cur_idx][2]])
	return appearance_triangles

# tmp, for collecting all mrce B common edges
def mrce_B_vis(tasks, r_name, idx_dict):
	s_map = {}
	for task in tasks:
		B = np.load('fs_results/0.1_mrce_B_'+task+'.npy')
		omega = B.T
		omega[omega!=0]=1
		s_sum = np.sum(omega, axis=0)

		print(task, np.count_nonzero(s_sum), s_sum.max(), s_sum.mean(), s_sum.min())
		sorted_idx_s = np.argsort(s_sum)[::-1]
		for i in range(np.count_nonzero(s_sum)):
			if sorted_idx_s[i] not in s_map:
				s_map[sorted_idx_s[i]] = s_sum[sorted_idx_s[i]]
			else:
				s_map[sorted_id_s[i]] += s_sum[sorted_idx_s[i]]
	print('\n')
	all_node_used = []
	for s_id, val in s_map.items():
		idx = idx_dict[s_id]
		if val > 50:
			print(r_name[idx[0]], r_name[idx[1]], val)
			print(idx[0], idx[1])
			if idx[0] not in all_node_used:
				all_node_used.append(idx[0])
			if idx[1] not in all_node_used:
				all_node_used.append(idx[1])
	import ipdb; ipdb.set_trace()

if __name__ == '__main__':
	with open('config.yaml') as info:
		info_dict = yaml.load(info)
	vis_dir = '/home/sikun/Downloads/BrainNetViewer/vis/'
	# task = sys.argv[1]
	tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']

	r_name = info_dict['data']['hcp']
	r = len(r_name)
	idx_dict = build_dict(r)

	mrce_B_vis(tasks, r_name, idx_dict)

	# inall = []
	# for task in tasks:
	# 	if task == 'resting':
	# 		r_name = info_dict['data']['aal']
	# 		omega = load_omega(task,mid='_',lam=0.1)
	# 	else:
	# 		r_name = info_dict['data']['hcp']
	# 		omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)
		
	# 	# appearance_nodes = common_nodes(omega, r_name, idx_dict)
	# 	appearance_edges = common_edges(omega, r_name, idx_dict, vis_dir)
	# 	# appearance_triangles = common_triangles(omega, r_name, idx_dict)
		
	# 	in_2core = np.asarray(appearance_edges)
	# 	in_2core[in_2core!=0] = 1
	# 	inall.append(in_2core)
	# inall = np.sum(np.stack(inall),axis=0)
	# print(np.where(inall==7)[0].shape)
