import numpy as np
from utility import cc_fista_inspect
from hcp_cc import data_prep
from f_recon import load_omega, regression

# check the relationship between single s and f edge
task = "LANGUAGE"
config_file = './config.yaml'
n_roi = 83
(sdata, fdata) = cc_fista_inspect.load_data(config_file, task)
for k in range(5):
    i = np.random.randint(low=0, high=n_roi-1)
    j = np.random.randint(low=i+1, high=n_roi-1)
    # cc_fista_inspect.check_single_edge(i, j, sdata, fdata)

# run regression for a single f edge
lambd = 0.0016
vec_s, vec_f = data_prep(task, v1=False, subj_ids=None)
omega = load_omega(task, mid='_er_train_hcp2_', lam=lambd)
fdir = 'fs_results/'
regression(vec_s, vec_f, omega, fdir, use_rnd=False, \
    use_train=True, lambd_values=[lambd], task=task)

# run regression in sklearn




