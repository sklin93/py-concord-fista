import numpy as np
from cvxpy import *

from hcp_cc import data_prep
from common_nonzero import load_omega

# load config, change config for different tasks
import yaml
with open('config.yaml') as info:
	info_dict = yaml.load(info)

vec_s, vec_f = data_prep(upenn=False)
n, p = vec_s.shape
task = info_dict['f_file'][6:-4]
# load omega
omega = load_omega(task)
k = np.count_nonzero(omega) # number of model parameters.

X = Variable((p,p))
constraints = [X[omega==0]==0]

pred_f = X.__matmul__(vec_s.T)
loss = norm(vec_f-pred_f.T)

lambd = 0.01
reg = norm(X, 2)
prob = Problem(Minimize(loss/n + lambd*reg),constraints)
print('prob set.')
prob.solve(solver=SCS)
import ipdb; ipdb.set_trace()

# compare with random positioned nz entries, with a same level sparsity