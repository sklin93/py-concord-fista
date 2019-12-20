from scipy.io import loadmat
import numpy as np
import pickle
import sys
from cc_fista import cc_fista, standardize, pseudol
from cc_mrce import mrce
from cscc_fista import cscc_fista
from hcp_cc import data_prep, f_pMat

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import time, yaml
import scipy.linalg as LA
import scipy.stats as st
from scipy.stats.stats import pearsonr
from cvxpy import *

from tqdm import tqdm
from sklearn import linear_model
from sklearn.metrics import mean_squared_error
from math import sqrt
import os.path

import sys
sys.path.append('./')
with open('config.yaml') as info:
    info_dict = yaml.load(info)

fdir = 'fs_results/'

def mrce_sf(task, b_lam=0.1, omega_lam=0.1, cross_val=10, load=False, 
			OUT_MAX_ITR=20, out_verbose=False, plot_verbose=False,
			eval_verbose=True, train_val=True, rnd_compare=False,
			check_common=False):

	def get_B_omega(vec_s, vec_f, pMat=None, b_lam=b_lam, omega_lam=omega_lam):
		p = vec_s.shape[1]
		q = vec_f.shape[1]
		B_hat   = np.zeros((p, q))
		Omg_hat = np.zeros((q, q))
		obj = float('inf')

		for i in range(OUT_MAX_ITR):
			print('Omg-B combined loop: ', i)
			# (estimate Omega)
			# partial correlation graph estimation
			problem = cscc_fista(D=vec_f-np.matmul(vec_s, B_hat),
						pMat=pMat, num_var=q,
						# step_type_out = args.cscc_step_type_out, const_ss_out = args.cscc_const_ss_out,
						# p_gamma=args.cscc_gamma, p_lambda=args.cscc_lambda, p_tau=args.cscc_tau,
						# MAX_ITR=args.cscc_max_itr,
						# TOL=args.cscc_TOL, TOL_inn=args.cscc_TOL_inn,
						# verbose=args.cscc_outer_verbose, verbose_inn=args.cscc_inner_verbose,
						# no_constraints=args.no_constraints, inner_cvx_solver=args.inner_cvx_solver,
						# record_label=record_label)
						p_lambda=omega_lam, 
						MAX_ITR=20, TOL=1e-3, TOL_inn=1e-2,
						verbose=out_verbose, verbose_inn=False, plot_in_loop=plot_verbose,
						no_constraints=False, inner_cvx_solver=False)
			Omg_hat, _ = problem.solver_convset()
			# (estimate B) 
			# regression coefficient estimation
			problem = mrce(Omg=np.linalg.matrix_power(Omg_hat,2),
						lamb2=b_lam, X=vec_s, Y=vec_f,
                        # step_type=args.mrce_step_type, const_ss=args.mrce_const_ss, p_tau=args.mrce_tau,
                        step_type=1, const_ss=0.1, p_tau=0.7,
                        c=0.5, alpha=1, TOL_ep=0.05, max_itr=50, 
                        # verbose=args.mrce_verbose, verbose_plots=args.mrce_verbose_plots)
                        verbose=out_verbose, verbose_plots=plot_verbose)
			B_hat   = problem.fista_solver_B()
			cur_obj = problem.likelihood_B(B_hat)
			print('objective at B: {:.3e}'.format(cur_obj))
			# stopping criterion
			# set stopping threshold: 1/1000 first obj value
			if i == 0: 
				thr = cur_obj / 1000
			if obj - cur_obj < thr:
				break
			obj = cur_obj
			# import ipdb; ipdb.set_trace()

		return B_hat, Omg_hat

	def recon_eval(vec_s, vec_f, B, Omg, b_lam):
		pred_f = np.matmul(vec_s, B)

		# recon error percentage wrt ground truth
		# err_percent = (LA.norm(vec_f - pred_f) ** 2) / (LA.norm(vec_f) ** 2)
		# err_percent = (vec_f - pred_f).sum() / vec_f.sum()
		err_percent = []

		# correlation coefficient and p-value
		avg_r = []
		min_pval = 1
		max_pval = 0
		for i in range(len(vec_s)):
			err_percent.append((vec_f[i] - pred_f[i]).sum() / vec_f[i].sum())
			r, pval = pearsonr(pred_f[i],vec_f[i])
			avg_r.append(r)
			if pval < min_pval:
				min_pval = pval
			if pval > max_pval:
				max_pval = pval
		err_percent = sum(err_percent)/len(err_percent)
		avg_r = sum(avg_r)/len(avg_r)
		# likelihood
		problem = mrce(Omg=np.linalg.matrix_power(Omg,2),
			lamb2=b_lam, X=vec_s, Y=vec_f,
            step_type=1, const_ss=0.1, p_tau=0.7,
            c=0.5, alpha=1, TOL_ep=0.05, max_itr=50,
            verbose=False, verbose_plots=False)
		obj = problem.likelihood_B(B)
		return err_percent, avg_r, min_pval, max_pval, obj


	vec_s, vec_f = data_prep(task, v1=True, normalize_s=True)
	if task == 'resting':
		pMat = f_pMat(90) 
	else:
		pMat = f_pMat(83) # nz 554689

	if cross_val == 0:
		b_name = fdir+str(b_lam)+'_mrce_B_'+task+'.npy'
		omg_name = fdir+str(omega_lam)+'_mrce_Omg_'+task+'.npy'
		if load and os.path.exists(b_name) and os.path.exists(omg_name):
			B = np.load(b_name)
			Omg = np.load(omg_name)
		else:
			B, Omg = get_B_omega(vec_s, vec_f, pMat=pMat)
			np.save(b_name, B)
			np.save(omg_name, Omg)
		print('B min:', B.min(), 'B max:', B.max(), 
				'B nz entry #: ', np.count_nonzero(B))		
		print('Omg min:', Omg.min(), 'Omg max:', Omg.max(), 
				'Omg nz entry #:', np.count_nonzero(Omg))

	else:
		# k-fold cross_val
		val_num = len(vec_s) // cross_val

		if train_val:
			train_err_avg	=	[]
			train_r_avg 	=	[]
			train_obj_avg 	=	[]
			train_p_min 	=	1
			train_p_max 	=	0

			val_err_avg	=	[]
			val_r_avg 	=	[]
			val_obj_avg =	[]
			val_p_min 	=	1
			val_p_max 	=	0

		if rnd_compare:
			rnd_err_avg	=	[]
			rnd_r_avg 	=	[]
			rnd_obj_avg =	[]
			rnd_p_min 	=	1
			rnd_p_max 	=	0

		if check_common:
			common_omg = []
			common_B = []
		for i in range(cross_val):
			print('cross validation, iteration:', i)
			val_s = vec_s[i * val_num: (i+1) * val_num]
			train_s = np.concatenate((vec_s[:i * val_num], vec_s[(i+1) * val_num:]))
			val_f = vec_f[i * val_num: (i+1) * val_num]
			train_f = np.concatenate((vec_f[:i * val_num], vec_f[(i+1) * val_num:]))

			b_name = fdir+str(b_lam)+'_mrce_B_'+task+str(i)+'.npy'
			omg_name = fdir+str(omega_lam)+'_mrce_Omg_'+task+str(i)+'.npy'
			if load and os.path.exists(b_name) and os.path.exists(omg_name):
				B = np.load(b_name)
				Omg = np.load(omg_name)
			else:
				B, Omg = get_B_omega(train_s, train_f, pMat=pMat)
				np.save(b_name, B)
				np.save(omg_name, Omg)
			
			if check_common:
				omg_nz = Omg.copy()
				omg_nz[omg_nz!=0] = 1
				common_omg.append(omg_nz)

				B_nz = B.copy()
				B_nz[B_nz!=0] = 1
				common_B.append(B_nz)

			if eval_verbose:
				# Stats of estimated B and Omega
				print('B min:', B.min(), 'B max:', B.max(), 
						'B nz entry #: ', np.count_nonzero(B))		
				print('Omg min:', Omg.min(), 'Omg max:', Omg.max(), 
						'Omg nz entry #:', np.count_nonzero(Omg))

			# evaluate
			if train_val:
				# training data stats (as baseline)
				err, avg_r, min_p, max_p, obj = recon_eval(train_s, train_f, B, Omg, b_lam)
				if eval_verbose:
					print('train recon err:', err, '; correlation coefficient:', avg_r, 
						'; min/max p value:', min_p, max_p, '; obj:', obj)
				train_err_avg.append(err)
				train_r_avg.append(avg_r)
				train_obj_avg.append(obj)
				if min_p < train_p_min:
					train_p_min = min_p
				if max_p > train_p_max:
					train_p_max = max_p
				# validation data stats
				err, avg_r, min_p, max_p, obj = recon_eval(val_s, val_f, B, Omg, b_lam)
				if eval_verbose:
					print('val recon err:', err, '; correlation coefficient:', avg_r, 
						'; min/max p value:', min_p, max_p, '; obj:', obj)
				val_err_avg.append(err)
				val_r_avg.append(avg_r)
				val_obj_avg.append(obj)
				if min_p < val_p_min:
					val_p_min = min_p
				if max_p > val_p_max:
					val_p_max = max_p

			if rnd_compare:
				# random comparison (random B & Omg w/ same sparsity level)
				p, q = B.shape
				rnd_B = np.zeros((p, q))
				idx = np.random.choice(p * q, np.count_nonzero(B), replace=False)
				ctr = 0
				for i in range(p):
					for j in range(q):
						if ctr in idx:
							rnd_B[i,j] = 1
						ctr += 1
				err, avg_r, min_p, max_p, obj = recon_eval(val_s, val_f, rnd_B, Omg, b_lam)
				if eval_verbose:
					print('val recon err with random B and Omega:', err, 
						'; correlation coefficient:', avg_r, 
						'; min/max p value:', min_p, max_p, '; obj:', obj)
				rnd_err_avg.append(err)
				rnd_r_avg.append(avg_r)
				rnd_obj_avg.append(obj)
				if min_p < rnd_p_min:
					rnd_p_min = min_p
				if max_p > rnd_p_max:
					rnd_p_max = max_p

		# check common B and Omg
		if check_common:
			common_B	=	np.sum(np.stack(common_B), axis=0)
			common_omg	=	np.sum(np.stack(common_omg), axis=0)

			p, q = common_B.shape
			B_nz_occr = {}
			omg_nz_occr = {}

			for i in range(p):
				for  j in range(q):
					if common_B[i, j] in B_nz_occr:
						B_nz_occr[common_B[i, j]] += 1
					else:
						B_nz_occr[common_B[i, j]] = 1
			for i in range(q):
				for  j in range(q):
					if common_omg[i, j] in omg_nz_occr:
						omg_nz_occr[common_omg[i, j]] += 1
					else:
						omg_nz_occr[common_omg[i, j]] = 1
			print('B nonzero occurrence across folds:')
			for i in sorted(B_nz_occr):
				print(i, ':', B_nz_occr[i])
			print('Omega nonzero occurrence across folds:')
			for i in sorted(omg_nz_occr):
				print(i, ':', omg_nz_occr[i])

			# save top common ones for visualization
			print('top common ratio in B:', np.sum(common_B >= cross_val - 2) / np.count_nonzero(common_B))
			common_B[common_B < cross_val - 2] = 0
			common_B[common_B != 0] = 1
			print('top common ratio in Omg:', np.sum(common_omg >= cross_val - 2) / np.count_nonzero(common_omg))
			common_omg[common_omg < cross_val - 2] = 0
			common_omg[common_omg != 0] = 1

			np.save(fdir+str(b_lam)+'_mrce_B_'+task+'_common_vis'+'.npy', common_B)
			np.save(fdir+str(omega_lam)+'_mrce_Omg_'+task+'_common_vis'+'.npy', common_omg)
			# import ipdb; ipdb.set_trace()
		# print average evaluation results for k-folds
		if train_val:
			train_err_avg	=	sum(train_err_avg)/len(train_err_avg)
			train_r_avg 	=	sum(train_r_avg)/len(train_r_avg)
			train_obj_avg	=	sum(train_obj_avg)/len(train_obj_avg)
			print('***TRAIN DATA***', cross_val, 'fold recon error average:', 
				train_err_avg, '; r average:', train_r_avg, 
				'; min/max p value:', train_p_min, train_p_max,
				'; average obj:', train_obj_avg)

			val_err_avg	=	sum(val_err_avg)/len(val_err_avg)
			val_r_avg 	=	sum(val_r_avg)/len(val_r_avg)
			val_obj_avg	=	sum(val_obj_avg)/len(val_obj_avg)
			print('***VAL DATA***', cross_val, 'fold recon error average:', 
				val_err_avg, '; r average:', val_r_avg, 
				'; min/max p value:', val_p_min, val_p_max,
				'; average obj:', val_obj_avg)

		if rnd_compare:
			rnd_err_avg	=	sum(rnd_err_avg)/len(rnd_err_avg)
			rnd_r_avg 	=	sum(rnd_r_avg)/len(rnd_r_avg)
			rnd_obj_avg	=	sum(rnd_obj_avg)/len(rnd_obj_avg)
			print('***VAL DATA WITH RAND B***', cross_val, 
				'fold recon error average:', rnd_err_avg, 
				'; r average:', rnd_r_avg, '; min/max p value:', 
				rnd_p_min, rnd_p_max, '; average obj:', rnd_obj_avg)

if __name__ == '__main__':
	task = sys.argv[1]
	b_lam = float(sys.argv[2])
	omega_lam = float(sys.argv[3])

	mrce_sf(task, b_lam=b_lam, omega_lam=omega_lam, load=True, eval_verbose=False,
		train_val=False, rnd_compare=False, check_common=True)
