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
# X = omega.copy()
# import ipdb; ipdb.set_trace()

# loss = 0
pred_f = X.__matmul__(vec_s.T)
loss = norm(vec_f-pred_f.T)
# for subj_idx in range(n):
# 	print(subj_idx)
# 	pred_f = X.__matmul__(vec_s[subj_idx].reshape(-1,1))
# 	loss += norm(vec_f[subj_idx] - pred_f)**2
# 	# cur_w_pos = 0
# 	# for f_idx in range(p):
# 	# 	cur_w_num = np.count_nonzero(omega[f_idx])
# 	# 	pred_f = sum(multiply(vec_s[subj_idx][omega[f_idx].astype(bool)],w[cur_w_pos:cur_w_pos+cur_w_num]))
# 	# 	cur_w_pos += cur_w_num
# 	# 	loss += norm(vec_f[subj_idx][f_idx] - pred_f)**2
lambd = 0.01
reg = norm(X, 2)
prob = Problem(Minimize(loss/n + lambd*reg),constraints)
print('prob set.')
prob.solve(solver=SCS)
import ipdb; ipdb.set_trace()

# compare with random positioned nz entries, with a same level sparsity