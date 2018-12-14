import numpy as np
import sys
from tqdm import tqdm
from cvxpy import *
from scipy.stats.stats import pearsonr

from hcp_cc import data_prep
from common_nonzero import load_omega, nz_share
from common_2core import common_edges
from vis import build_dict

task = sys.argv[1]
fdir = 'fs_results/'
vec_s, vec_f = data_prep(task)
n, p = vec_s.shape
# load omega
if task == 'resting':
	omega = load_omega(task,mid='_',lam=0.1)
else:
	omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)

def p_val(result):
	pred_f = vec_s@result.T
	print(pred_f.shape)
	pval_tot = 0
	for k in range(n):
		cor, pval = pearsonr(pred_f[k],vec_f[k])
		print(cor, pval)
		pval_tot += pval
	return pval_tot/n

check_only = True
use_rnd = False
''' 
# direct regression (OOM)

X = Variable((p,p))

def loss_fn(X, vec_s, vec_f):
    return pnorm(matmul(vec_s, X.T) - vec_f, p=2)**2

def regularizer(X):
    return pnorm(X, p=2)**2
def objective_fn(X, vec_s, vec_f, lambd):
    return loss_fn(X, vec_s, vec_f) + lambd * regularizer(X)


def regularizer2(X, lamMat):
	return sum(multiply(lamMat,power(X,2)))
def objective_fn2(X, vec_s, vec_f, lamMat):
	return loss_fn(X, vec_s, vec_f) + regularizer2(X,lamMat)

lambd = 0.01
hard_constraint = False
if hard_constraint:
	constraints = [X[omega==0]==0]
	prob = Problem(Minimize(objective_fn(X,vec_s,vec_f,lambd)),constraints)
else:
	lamMat = np.ones((p,p))*lambd
	lamMat[omega==0] *= 10000
	prob = Problem(Minimize(objective_fn2(X,vec_s,vec_f,lamMat)))

print('Problem set.')
prob.solve()
'''

# do regression row by row

def loss_fn(w, s, f):
	return norm(matmul(s, w) - f)**2
def objective_fn(w, s, f, lam):
	return loss_fn(w, s, f) + lam * norm(w)**2

def objective_fn2(w, s, f, lam, vec):
	return loss_fn(w, s, f) + lam * norm(vec*w)**2

if check_only:
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
else:
	if use_rnd:
		# compare with random positioned nz entries, with a same level sparsity
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
	
	s = vec_s
	# lambd_values = np.logspace(-1, 1, 10)
	lambd_values = [0.33]
	hard_constraint = True
	# pval_values = []
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
		# pval_values.append(cur_pval)
		'''
		# check if results all fall in nonzero positions (ratio should be 1)
		tmp = result.copy()
		tmp[tmp!=0]=1
		ratio, _ = nz_share(tmp,omega)
		print(ratio)
		'''