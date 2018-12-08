import numpy as np
import sys
from cvxpy import *

from hcp_cc import data_prep
from common_nonzero import load_omega

task = sys.argv[1]
vec_s, vec_f = data_prep(task)
n, p = vec_s.shape
# load omega
omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)
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

print('prob set.')
prob.solve()

# import ipdb; ipdb.set_trace()
# do regression row by ro
# compare with random positioned nz entries, with a same level sparsity