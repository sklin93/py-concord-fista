import numpy as np
import sys
from cvxpy import *

from hcp_cc import data_prep
from common_nonzero import load_omega

# load config, change config for different tasks
import yaml
with open('config.yaml') as info:
	info_dict = yaml.load(info)

task = sys.argv[1]
vec_s, vec_f = data_prep(task)
n, p = vec_s.shape
# load omega
omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)
# k = np.count_nonzero(omega) # number of model parameters.

X = Variable((p,p))
constraints = [X[omega==0]==0]

def loss_fn(X, vec_s, vec_f):
    return pnorm(matmul(vec_s, X.T) - vec_f, p=2)**2

def regularizer(X):
    return pnorm(X, p=2)**2

def objective_fn(X, vec_s, vec_f, lambd):
    return loss_fn(X, vec_s, vec_f) + lambd * regularizer(X)


lambd = 0.01
prob = Problem(Minimize(objective_fn(X,vec_s,vec_f,lambd)),constraints)
print('prob set.')
prob.solve()
import ipdb; ipdb.set_trace()

# compare with random positioned nz entries, with a same level sparsity