import numpy as np
import sys
from tqdm import tqdm
from cvxpy import *
from scipy.stats.stats import pearsonr

from hcp_cc import data_prep
from common_nonzero import load_omega, nz_share

task = sys.argv[1]
fdir = 'fs_results/'
vec_s, vec_f = data_prep(task)
n, p = vec_s.shape
# load omega
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

check_only = False
use_rnd = False
'''
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
	print(p_val(result))
else:
	if use_rnd:
		# compare with random positioned nz entries, with a same level sparsity
		rnd_w = np.zeros(omega.shape)
		idx = np.random.choice(d*d,np.count_nonzero(omega),replace=False)
		ctr = 0
		for i in range(d):
			for j in range(d):
				if ctr in idx:
					rnd_w[i,j] = 1
				ctr += 1
		print(np.count_nonzero(rnd_w))
		omega = rnd_w
	
	s = vec_s
	lambd_values = np.logspace(-1, 0, 10)
	hard_constraint = True
	pval_values = []
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
		# np.save(fdir+'weights_'+task+'.npy', result)
		cur_pval = p_val(result)
		print(lambd, np.count_nonzero(result), cur_pval)
		pval_values.append(cur_pval)
		'''
		# check if results all fall in nonzero positions (ratio should be 1)
		tmp = result.copy()
		tmp[tmp!=0]=1
		ratio, _ = nz_share(tmp,omega)
		print(ratio)
		'''