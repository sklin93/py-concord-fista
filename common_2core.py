import numpy as np 
import networkx as nx
import sys
from hcp_cc import data_prep
from vis import build_dict, get_graph
import yaml
with open('config.yaml') as info:
    info_dict = yaml.load(info)

fdir = 'fs_results/'
vis_dir = '/home/sikun/Downloads/BrainNetViewer/vis/'
task = sys.argv[1]
if task == 'resting':
	r_name = info_dict['data']['aal']
else:
	r_name = info_dict['data']['hcp']
r = len(r_name)
omega = np.load(fdir+'0.0014_1stage_er2_'+task+'.npy') #p*2p
omega = omega[:,omega.shape[0]:]
print(omega.shape)
print(np.count_nonzero(omega))

omega[omega!=0] = 1
idx_dict = build_dict(r)

# common nodes:
'''
appearance_nodes = np.zeros(r)
for i in range(omega.shape[0]):
	G = get_graph(omega, r, idx_dict, i)
	subG = nx.k_core(G,2)
	cur_nodes = list(subG.nodes)
	appearance_nodes[cur_nodes] += 1

size = appearance_nodes/appearance_nodes.max()
np.save('size.npy',size)

sorted_idx = np.argsort(appearance_nodes)[::-1]
for i in range(r):
	cur_idx = sorted_idx[i]
	if appearance_nodes[cur_idx]!=0:
		print(appearance_nodes[cur_idx], r_name[cur_idx])
'''

# common edges:
'''
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
np.savetxt(vis_dir+'common_edge.edge', appearance_edges, fmt='%i')
'''

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
		

