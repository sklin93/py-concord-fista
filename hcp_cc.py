from scipy.io import loadmat
import numpy as np
import sys
from cc_fista import cc_fista, standardize, pseudol
import time, yaml
from scipy.linalg import norm
from scipy.stats.stats import pearsonr

with open('config.yaml') as info:
    info_dict = yaml.load(info)

fdir = 'fs_results/'

def data_prep(task, normalize_s=False):
	if task=='resting':
		dataMat = loadmat(info_dict['data_dir_Bassette']+info_dict['Bassette_file'])
		sMat = dataMat['Ss']
		fMat = dataMat['Fs']
		s = []
		f = []
		for i in range(sMat.shape[0]):
			s.append(sMat[i][0])
			f.append(fMat[i][0])
		s = np.stack(s, axis=2)
		f = np.stack(f, axis=2)
	else:
		sMat = loadmat(info_dict['data_dir']+info_dict['s_file'])
		s = sMat['X']
		# fMat = loadmat(info_dict['data_dir']+info_dict['f_file'])
		fMat = loadmat(info_dict['data_dir']+'tfMRI-'+task+'.mat')
		f = fMat['X']
	 
	d,_,n = s.shape
	vec_s = []
	vec_f = []
	p = 0
	for i in range(d-1):
	  for j in range(i+1,d):
	      vec_s.append(s[i,j])
	      vec_f.append(f[i,j])
	      p = p+1
	vec_s = np.transpose(np.asarray(vec_s))
	vec_f = np.transpose(np.asarray(vec_f))
	if normalize_s:
		vec_s -= vec_s.min()
		vec_s /= vec_s.max()
	assert vec_s.shape == vec_f.shape, 'F and S size mismatch.'
	return vec_s, vec_f

def s_f_pMat(d): # d is p/2
	tmp_14 = np.identity(d)*2
	tmp_23 = np.ones((d,d))
	np.fill_diagonal(tmp_23, 2)
	up = np.concatenate((tmp_14,tmp_23),axis=1)
	low = np.concatenate((tmp_23,tmp_14),axis=1)
	return np.concatenate((up,low),axis=0)

def f_only(task, lam):
	# original version
	_, vec = data_prep(task)
	fi = cc_fista(vec,lam,steptype=1)
	print('Input vector shape: ', vec.shape)
	start = time.time()
	invcov = fi.infer()
	print((time.time()-start)/60)
	print(np.count_nonzero(invcov))
	print(np.count_nonzero(invcov.diagonal()))
	print(fi.loss())

def s_f(task, lam, check_loss_only=False, split=False):
	vec_s, vec_f = data_prep(task)
	if split:
		train_num = int(vec_s.shape[0]*0.8)
		vec_s = vec_s[:train_num,:]
		vec_f = vec_f[:train_num,:]
	vec = np.concatenate((vec_f,vec_s), axis=1)
	print('Input vector shape: ', vec.shape)
	if check_loss_only:
		check_loss(vec,np.load(fdir+str(lam)+task+'.npy'))
		return
	fi = cc_fista(vec, lam, s_f=True, steptype=3, const_ss=0.8)
	# fi = cc_fista(vec, lam, s_f=True)
	start = time.time()
	omega = fi.infer()
	print((time.time()-start)/60)
	if split:
		np.save(fdir+str(lam)+'_er_train_'+task+'.npy',omega)
	else:
		np.save(fdir+str(lam)+'_1stage_er2_'+task+'.npy',omega)
	print(np.count_nonzero(omega))
	d = omega.shape[0]
	print(np.count_nonzero(omega[:,:d]))
	print(np.count_nonzero(omega[:,d:]))
	print(np.count_nonzero(omega[:,:d].diagonal()))
	print(np.count_nonzero(omega[:,d:].diagonal()))
	print(fi.loss())
	# import ipdb; ipdb.set_trace()

def s_f_direct(task,lam):
	vec_s, vec_f = data_prep(task)
	vec = np.concatenate((vec_f,vec_s), axis=1)
	print('Input vector shape: ', vec.shape)
	pMat = s_f_pMat(int(vec.shape[1]/2))
	fi = cc_fista(vec, lam, pMat=pMat)
	start = time.time()
	omega = fi.infer()
	print((time.time()-start)/60)
	np.save(fdir+'direct_'+str(lam)+'.npy',omega)

def check_loss(D,X):
	print(np.count_nonzero(X))
	d = X.shape[0]
	print(np.count_nonzero(X[:,:d]))
	print(np.count_nonzero(X[:,d:]))
	print(np.count_nonzero(X[:,:d].diagonal()))
	print(np.count_nonzero(X[:,d:].diagonal()))
	S = standardize(D)
	print(pseudol(X,S@X.transpose()))

def reconstruct_err(task, filename, rnd_compare=False):
	vec_s, vec_f = data_prep(task)
	n = vec_s.shape[0]
	omega = np.load(filename)
	d = int(omega.shape[1]/2)

	# vec = np.concatenate((vec_f,vec_s), axis=1)
	# S = standardize(vec)
	# print((omega.transpose()*(S@omega.transpose())).sum())

	omega = omega[:d,d:]
	print(omega.min(),omega.mean(),omega.max())
	print(omega.shape)
	# rho = np.zeros(omega.shape)
	# import ipdb; ipdb.set_trace()
	# for i in range(d):
	# 	for j in range(d):
	# 		if i==j:
	# 			rho[i,j] = 1
	# 		else:
	# 			rho[i,j] = -omega[i,j]/np.sqrt(omega[i,i]*omega[j,j])
	print(np.count_nonzero(omega))
	pred_f = vec_s@omega.T
	print(pred_f.shape)
	for k in range(n):
		print(pearsonr(pred_f[k],vec_f[k]))

	if rnd_compare:
		# create random 'omega' with a same level of sparsity
		rnd_w = np.zeros(omega.shape)
		idx = np.random.choice(d*d,np.count_nonzero(omega),replace=False)
		ctr = 0
		for i in range(d):
			for j in range(d):
				if ctr in idx:
					rnd_w[i,j] = 1
				ctr += 1
		print(np.count_nonzero(rnd_w))
		pred_f_rnd = vec_s@rnd_w.T
		print(pred_f_rnd.shape)
		for k in range(n):
			print(pearsonr(pred_f_rnd[k],vec_f[k]))
		
	# err = tot_err_perct = 0.0
	# for k in range(n):
	# 	cur_s = vec_s[k]
	# 	f_gt = vec_f[k]
	# 	f_model = np.zeros(d)
	# 	for i in range(d):
	# 		f_model[i] = - (omega[i].dot(cur_s))/omega[i,i]
	# 	print(f_model.min(),f_model.mean(),f_model.max())
	# 	# cur_err = norm(f_model-f_gt)
	# 	# err_perct = cur_err/norm(f_gt)
	# 	# print(err_perct)
	# 	print(pearsonr(f_model,f_gt))
		# tot_err_perct += err_perct
		# err += cur_err
	# print('average error percentage',tot_err_perct/n)
	# return err


if __name__ == '__main__':
	task = sys.argv[1]
	# f_only(task,lam=0.1)
	s_f(task,lam=0.0014, check_loss_only=False, split=True) # use 0.0012 for normalization 2
	# s_f_direct(task,lam=0.08)
	# reconstruct_err(task,fdir+'0.0014_1stage_er2_'+task+'.npy')
