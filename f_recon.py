import numpy as np
import sys
from tqdm import tqdm
from cvxpy import *
from scipy.stats.stats import pearsonr
from sklearn.metrics import mean_squared_error
from math import sqrt
import pickle
from scipy.io import loadmat

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
		print('cc, pval: ', cor, pval)
		print('rms: ', rms)
		pval_tot += pval
		rms_tot += rms
	print('summary: ', pval_tot/n, rms_tot/n)
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

	# load data
	# hcp v2 test set ids
	subj_ids = ['199655', '188347', '153025', '124220', '154431', '168341', '153429', '199150', '211720', '169343', '120212', '212419', '158136', '123420', '149741', '200614', '211215', '150726', '200109', '163836', '159441', '194645', '158035', '173334', '151627', '214726', '246133', '122317', '145834', '155231', '154734', '133928', '179548', '156233', '212217', '144832', '149337', '175035', '212318', '195041', '151223', '148941']
	# hcp v2 train set ids
	# subj_ids = ['195849', '147030', '123925', '160123', '120111', '189349', '136227', '172029', '250427', '171633', '159239', '198451', '157336', '156637', '131217', '118730', '181131', '178950', '121618', '133625', '205826', '211922', '157437', '201111', '155635', '133019', '118528', '164030', '214019', '135225', '136833', '205725', '185442', '147737', '208226', '144226', '135932', '164939', '133827', '151728', '205119', '161731', '172332', '186141', '177645', '245333', '165840', '146432', '214221', '135528', '210617', '143325', '122620', '140925', '187850', '137633', '178142', '173536', '166438', '192843', '148032', '231928', '255639', '146331', '167036', '204521', '250932', '163331', '180836', '123117', '221319', '158540', '173435', '199958', '172938', '199453', '141422', '138534', '120515', '162733', '256540', '217429', '211316', '180432', '130316', '161630', '224022', '196750', '196144', '138231', '177746', '205220', '162329', '154936', '208327', '165032', '126325', '141826', '176542', '139637', '175439', '190031', '127933', '180129', '191841', '128127', '181232', '148840', '169444', '210011', '182739', '201818', '191033', '192540', '162026', '152831', '212116', '163129', '251833', '214423', '127630', '204016', '118932', '139233', '173940', '149539', '179346', '201414', '203418', '126628', '187547', '131722', '130922', '140117', '194847', '150625', '180937', '191437', '189450', '211417', '148335', '119833', '151526', '185139', '130013', '128632', '162228', '161327', '159340', '217126', '131924', '140824', '191336', '140420', '239944', '124826', '178849', '182840', '154835', '124422', '197348', '193239', '194140', '172130', '160830', '233326', '129028']
	# vec_s, vec_f = data_prep(task, v1=False, subj_ids=subj_ids)
	vec_s, vec_f = data_prep(task)
	# vec_s, vec_f = vec_s[:14], vec_f[:14]
	
	'''
	# load omega
	if task == 'resting':
		omega = load_omega(task,mid='_',lam=0.1)
	else:
		# omega = load_omega(task,mid='_1stage_er2_',lam=0.0014)
		omega = load_omega(task,mid='_er_train_',lam=0.0014)
	fdir = 'fs_results/'
	regression(vec_s, vec_f, omega, fdir, use_rnd=True, use_train=True, lambd_values=[lambd])
	# result_check(fdir, task, omega)
	'''
	# load pred_f, should be in shape n*p
	pred_f = np.load('./cgmm_results/pred_f_CD_0.005.npy')
	# pred_f = np.load('pred_f_vae_train.npy')
	# pred_f = loadmat('pred_f_spectral_language_grp.mat')
	# pred_f = pred_f['pred_f']
	# print(pred_f.shape)
	# import ipdb; ipdb.set_trace()
	# # ''' uncomment if pred_f in shape n*r*r
	# n,r,_ = pred_f.shape
	# pred_f_ = np.zeros((n,int(r*(r-1)/2)))
	
	# p = 0
	# for k in range(n):
	# 	p = 0
	# 	for i in range(r-1):
	# 		for j in range(i+1,r):
	# 			pred_f_[k,p] = pred_f[k,i,j]
	# 			p += 1
	# pred_f = pred_f_
	# '''
	evaluate(vec_s, vec_f, pred_f, is_pred=True)