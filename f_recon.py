import numpy as np
import sys
from tqdm import tqdm
from cvxpy import *
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt

from hcp_cc import data_prep
from common_nonzero import load_omega, nz_share
from common_2core import common_edges
from vis import build_dict

def evaluate(vec_s, vec_f, result, is_pred=False):
	''' evaluate prediction based on p value and root mean squared error.
	if is_pred is True, then the input result is pred_f
	if is_pred is False (default), then the input result is regressed weights'''
	if is_pred:
		pred_f = result
	else:
		pred_f = vec_s@result.T
	print(pred_f.shape)
	n = pred_f.shape[0]
	pval_tot = 0
	rms_tot = 0
	for k in range(n):
		cor, pval = pearsonr(pred_f[k],vec_f[k])
		rms = sqrt(mean_squared_error(vec_f[k], pred_f[k]))
		print(cor, pval)
		print(rms)
		pval_tot += pval
		rms_tot += rms
	return pval_tot/n, rms_tot/n

def loss_fn(w, s, f):
	return norm(matmul(s, w) - f)**2

def objective_fn(w, s, f, lam):
	return loss_fn(w, s, f) + lam * norm(w)**2

def objective_fn2(w, s, f, lam, vec):
	return loss_fn(w, s, f) + lam * norm(vec*w)**2

def result_check(fdir, task, omega):
	result = np.load(fdir+'weights_'+task+'.npy')
	print(np.count_nonzero(result))
	print(evaluate(result))
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

def regression(vec_s, vec_f, omega, fdir, lambd_values=None, use_rnd=False, 
			use_train=False, hard_constraint=True, check_constraint=False):
	if use_train:
		train_num = int(vec_s.shape[0]*0.8)
		train_s = vec_s[:train_num,:]
		train_f = vec_f[:train_num,:]	
	else:
		train_s = vec_s
		train_f = vec_f
	p = vec_s.shape[1]
	if use_rnd:
		rnd_w = np.zeros(omega.shape)
		nz_num = np.count_nonzero(omega)
		# compare with random positioned nz entries, with a same level sparsity
		'''
		idx = np.random.choice(p*p, nz_num, replace=False)
		ctr = 0
		for i in range(p):
			for j in range(p):
				if ctr in idx:
					rnd_w[i,j] = 1
				ctr += 1
		'''
		# compare with chosen common edges (appears in 2-core across 7 tasks)
		# '''
		inall = list(np.load('edge_choice.npy'))
		col_num = len(inall)
		print('chosen edge #: ', col_num)
		idx = np.random.choice(col_num*p, nz_num, replace=False)
		ctr = 0
		for i in range(p):
			for j in range(p):
				if j in inall:
					if ctr in idx:
						rnd_w[i,j] = 1
					ctr += 1
		print(np.count_nonzero(np.sum(rnd_w,axis=0)))
		# '''
		print(np.count_nonzero(rnd_w))
		print((nz_num+np.count_nonzero(rnd_w)-
					np.count_nonzero(omega-rnd_w))/2)
		omega = rnd_w
	
	if lambd_values is None:
		lambd_values = np.logspace(-1, 1, 10)
	_min = 1000000
	for lambd in lambd_values:
		result = []
		for f_idx in range(p):
			w = Variable(p)
			f = train_f[:,f_idx]
			if hard_constraint:
				constraints = [w[omega[f_idx]==0] == 0]
				prob = Problem(Minimize(objective_fn(w, train_s, f, lambd)),constraints)
			else:
				vec = np.ones(p)
				vec[omega[f_idx]==0] *= 1e15
				prob = Problem(Minimize(objective_fn2(w, train_s, f, lambd, vec)))
			prob.solve()
			# if prob.status not in ["infeasible", "unbounded"]:
			# 	print("Optimal value: %s" % prob.value)
			# for variable in prob.variables():
			# 	print("Variable %s: value %s" % (variable.name(), variable.value))
			result.append(w.value)
		result = np.stack(result)
		result[result<1e-5] = 0

		if use_train:
			val_s = vec_s[train_num:,:]
			val_f = vec_f[train_num:,:]
		else:
			val_s = vec_s
			val_f = vec_f
		cur_pval, cur_rmse = evaluate(val_s, val_f, result)
		# save the best
		if cur_rmse < _min:
			if use_train:
				np.save(fdir+'weights_train_'+task+'.npy', result)
			else:
				np.save(fdir+'weights_'+task+'.npy', result)
			_min = cur_rmse
		print(lambd, np.count_nonzero(result), cur_pval, cur_rmse)
		if check_constraint:
			# check if results all fall in nonzero positions (ratio should be 1)
			tmp = result.copy()
			tmp[tmp!=0]=1
			ratio, _ = nz_share(tmp,omega)
			print(ratio)

if __name__ == '__main__':
	if len(sys.argv) == 2:
		task = sys.argv[1]
	elif len(sys.argv) == 3:
		task = sys.argv[1]
		lambd = sys.argv[2]

	fdir = 'fs_results/'
	# load data
	vec_s, vec_f = data_prep(task)
	'''
	# load omega
	if task == 'resting':
		omega = load_omega(task,mid='_',lam=0.1)
	else:
		# omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)
		omega = load_omega(task,mid='_er_train_',lam=0.0014)

	regression(vec_s, vec_f, omega, fdir, use_rnd=True, use_train=True, lambd_values=[lambd])
	# result_check(fdir, task, omega)
	'''
	# load pred_f
	pred_f = np.load('pred_f_BCD.npy')
	evaluate(vec_s, vec_f, pred_f, is_pred=True)