import numpy as np
import sys
from tqdm import tqdm
from cvxpy import *
from scipy.stats.stats import pearsonr

from hcp_cc import data_prep
from common_nonzero import load_omega, nz_share
from common_2core import common_edges
from vis import build_dict

def p_val(result):
	pred_f = vec_s@result.T
	print(pred_f.shape)
	n = pred_f.shape[0]
	pval_tot = 0
	for k in range(n):
		cor, pval = pearsonr(pred_f[k],vec_f[k])
		print(cor, pval)
		pval_tot += pval
	return pval_tot/n

def loss_fn(w, s, f):
	return norm(matmul(s, w) - f)**2

def objective_fn(w, s, f, lam):
	return loss_fn(w, s, f) + lam * norm(w)**2

def objective_fn2(w, s, f, lam, vec):
	return loss_fn(w, s, f) + lam * norm(vec*w)**2

def result_check(fdir, task, omega):
	result = np.load(fdir+'weights_'+task+'.npy')
	print(np.count_nonzero(result))
	print(p_val(result))
	import yaml
	with open('config.yaml') as info:
		info_dict = yaml.load(info)
	if task == 'resting':
		r_name = info_dict['data']['aal']
	else:
		r_name = info_dict['data']['hcp']
	r = len(r_name)
	idx_dict = build_dict(r)

	edge = common_edges(omega, r_name, idx_dict, save=False)

	sorted_idx = np.unravel_index(result.argsort(axis=None)[::-1],result.shape)
	topk = 20
	ctr = 0
	for i in range(len(sorted_idx[0])):
		if ctr >= topk:
			break
		if sorted_idx[0][i] != sorted_idx[1][i]:
			print(sorted_idx[0][i], sorted_idx[1][i], result[sorted_idx[0][i], sorted_idx[1][i]])
			print(r_name[idx_dict[sorted_idx[1][i]][0]], r_name[idx_dict[sorted_idx[1][i]][1]])
			ctr += 1

def regression(s, vec_f, omega, fdir, use_rnd=False, use_train=False,
				hard_constraint=True, check_constraint=False):
	if use_train:
		train_num = int(s.shape[0]*0.8)
		s = s[:train_num,:]
		vec_f = vec_f[:train_num,:]	
	n, p = s.shape
	if use_rnd:
		# compare with random positioned nz entries, with a same level sparsity
		#TODO: set diag to nz and others rnd
		rnd_w = np.zeros(omega.shape)
		idx = np.random.choice(p*p,np.count_nonzero(omega),replace=False)
		ctr = 0
		for i in range(p):
			for j in range(p):
				if ctr in idx:
					rnd_w[i,j] = 1
				ctr += 1
		print(np.count_nonzero(rnd_w))
		omega = rnd_w
	
	# lambd_values = np.logspace(-1, 1, 10)
	lambd_values = [0.33]
	_min = 1
	for lambd in lambd_values:
		result = []
		for f_idx in range(p):
			w = Variable(p)
			f = vec_f[:,f_idx]
			if hard_constraint:
				constraints = [w[omega[f_idx]==0] == 0]
				prob = Problem(Minimize(objective_fn(w, s, f, lambd)),constraints)
			else:
				vec = np.ones(p)
				vec[omega[f_idx]==0] *= 1e15
				prob = Problem(Minimize(objective_fn2(w, s, f, lambd, vec)))

			prob.solve()
			# if prob.status not in ["infeasible", "unbounded"]:
			# 	print("Optimal value: %s" % prob.value)
			# for variable in prob.variables():
			# 	print("Variable %s: value %s" % (variable.name(), variable.value))
			result.append(w.value)

		result = np.stack(result)
		result[result<1e-5] = 0
		cur_pval = p_val(result)
		# save the best
		if cur_pval < _min:
			np.save(fdir+'weights_'+task+'.npy', result)
			_min = cur_pval
		print(lambd, np.count_nonzero(result), cur_pval)
		if check_constraint:
			# check if results all fall in nonzero positions (ratio should be 1)
			tmp = result.copy()
			tmp[tmp!=0]=1
			ratio, _ = nz_share(tmp,omega)
			print(ratio)

if __name__ == '__main__':
	task = sys.argv[1]
	fdir = 'fs_results/'
	# load data
	vec_s, vec_f = data_prep(task)
	# load omega
	if task == 'resting':
		omega = load_omega(task,mid='_',lam=0.1)
	else:
		omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)

	use_train = True

	regression(vec_s, vec_f, omega, fdir, use_rnd=False, use_train=True)
	# result_check(fdir, task, omega)