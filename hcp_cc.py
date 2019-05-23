from scipy.io import loadmat
import numpy as np
import pickle
import sys
from cc_fista import cc_fista, standardize, pseudol
import time, yaml
from scipy.linalg import norm
from scipy.stats.stats import pearsonr
from cvxpy import *

with open('config.yaml') as info:
    info_dict = yaml.load(info)

fdir = 'fs_results/'

def data_prep(task, v1=True, subj_ids=None, normalize_s=False):
	if task == 'syn_sf':
		with open('data-utility/syn_sf.pkl','rb') as f: 
			data = pickle.load(f)
			vec_s = data['S']
			vec_f = data['F']

	elif task == 'resting':
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
		if v1:
			sMat = loadmat(info_dict['data_dir']+info_dict['s_file'])
			s = sMat['X']
			# fMat = loadmat(info_dict['data_dir']+info_dict['f_file'])
			fMat = loadmat(info_dict['data_dir']+'tfMRI-'+task+'.mat')
			f = fMat['X']
		else:
			with open(info_dict['data_dir']+info_dict['s_file'], 'rb') as f:
				sdata = pickle.load(f)
			with open(info_dict['data_dir']+'corrmats_tfMRI_'+task+'_125mm_LR_ROI_scale33.p','rb') as f:
				fdata = pickle.load(f, encoding='latin1')
			# if subject ids are not specified, then load all the subject data
			if subj_ids == None:
				subj_ids = []
				for k in sdata:
					subj_ids.append(k)
			s = []
			f = []
			for subj_id in subj_ids:
				s.append(sdata[subj_id])
				f.append(fdata[subj_id])
			s = np.stack(s, axis=2)
			f = np.stack(f, axis=2)

	if task != 'syn_sf':
		d,_,n = s.shape
		vec_s = []
		vec_f = []
		# p = 0
		for i in range(d-1):
			for j in range(i+1,d):
				vec_s.append(s[i,j])
				vec_f.append(f[i,j])
				# p = p+1
		vec_s = np.transpose(np.asarray(vec_s))
		vec_f = np.transpose(np.asarray(vec_f))
	if normalize_s:
		vec_s -= vec_s.min()
		vec_s /= vec_s.max()
	assert vec_s.shape == vec_f.shape, 'F and S size mismatch.'
	return vec_s, vec_f

def s_f_pMat(d): # d is p/2
	'''
	# mask out SS and FF (only keep diagonal)
	tmp_14 = np.identity(d)*2
	tmp_23 = np.ones((d,d))
	np.fill_diagonal(tmp_23, 2)
	up = np.concatenate((tmp_14,tmp_23),axis=1)
	low = np.concatenate((tmp_23,tmp_14),axis=1)
	'''
	# mask out SS (only keep diagonal)
	tmp_1 = np.ones((d,d))
	np.fill_diagonal(tmp_1, 2)
	tmp_23 = np.ones((d,d))
	tmp_4 = np.identity(d)*2
	up = np.concatenate((tmp_1,tmp_23),axis=1)
	low = np.concatenate((tmp_23,tmp_4),axis=1)
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
	vec_s, vec_f = data_prep(task, v1=False)
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
	# fi = cc_fista(vec, lam, s_f=True, steptype=2) # steptype2 blow
	start = time.time()
	omega = fi.infer()
	print((time.time()-start)/60)
	if split:
		if task == 'resting' or task == 'syn_sf':
			np.save(fdir+str(lam)+'_train_'+task+'.npy',omega)
		else:
			np.save(fdir+str(lam)+'_er_train_hcp2_'+task+'.npy',omega)
	else:
		if task == 'resting' or task == 'syn_sf':
			np.save(fdir+str(lam)+'_'+task+'.npy',omega)
		else:
			np.save(fdir+str(lam)+'_er_hcp2_'+task+'.npy',omega)
	print(np.count_nonzero(omega))
	d = omega.shape[0]
	print(np.count_nonzero(omega[:,:d]))
	print(np.count_nonzero(omega[:,d:]))
	print(np.count_nonzero(omega[:,:d].diagonal()))
	print(np.count_nonzero(omega[:,d:].diagonal()))
	print(fi.loss())

def s_f_direct(task, lam, split=False):
	'''directly perform concord on p*p matrix instead of d*p'''
	vec_s, vec_f = data_prep(task, v1=False)
	if split:
		train_num = int(vec_s.shape[0]*0.8)
		vec_s = vec_s[:train_num,:]
		vec_f = vec_f[:train_num,:]
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

"""Other packaged methods"""
'''Using sgcrf package to solve the problem'''
def sgcrf(task):
	import sys
	sys.path.insert(0,'/home/sikun/Documents/sgcrfpy/')
	from sgcrf import SparseGaussianCRF

	vec_s, vec_f = data_prep(task)
	sgcrf = SparseGaussianCRF(learning_rate=0.1)
	sgcrf.fit(vec_s, vec_f)
	# loss = sgcrf.lnll
	pred_f = sgcrf.predict(vec_s)
	for k in range(vec_s.shape[0]):
		print(pearsonr(pred_f[k],vec_f[k]))

'''Using cvxpy to solve CONCORD objective function'''
def regularizer(W, lam, pMat):
	p = W.shape[0]
	LambdaMat = lam*np.ones((p,p))
	np.fill_diagonal(LambdaMat, 0)
	if pMat is not None:
		""" pMat: entry==0 means result here must be 0, 
		entry==2 means results here must be nonzero,
		entry==1 is a normal entry """
		LambdaMat[pMat==0] *= 10000
		LambdaMat[pMat==2] = 0
	return norm(multiply(LambdaMat, W), 1)
	
def cc_obj(S, W, lam, pMat):
	# pseudol_ = -0.5*log_det(diag(diag(W)**2)) + 0.5*sum((W.T*(S@W))) (0.5*trace(W@S@W))
	pseudol_ = -sum(log(diag(W))) + 0.5*sum([quad_form(W[i,:], S) for i in range(S.shape[0])])
	return pseudol_ + regularizer(W, lam, pMat)

def cc_cvx(task, lam, pMat=None, s_f=False, split=False):
	if task == 'syn':
		# load syn data
		with open('./data-utility/syn.pkl', 'rb') as f:
			(Omg, Sig, vec, pMat, num_var, num_smp) = pickle.load(f)
	else:
		# load MRI data
		vec_s, vec_f = data_prep(task, v1=False)
		if split:
			train_num = int(vec_s.shape[0]*0.8)
			vec_s = vec_s[:train_num,:]
			vec_f = vec_f[:train_num,:]
		if s_f:
			vec = np.concatenate((vec_f,vec_s), axis=1)
		else:
			vec = vec_f
	print('Input vector shape: ', vec.shape)
	S = standardize(vec)
	p = S.shape[0]
	W = Variable((p,p))
	prob = Problem(Minimize(cc_obj(S, W, lam, pMat)))
	prob.solve(CVXOPT, verbose=True)
	result = W.value
	result[np.where(result < 1e-5)] = 0
	if task == 'syn':
		print('omega:\n', np.round(Omg,3))
	print('cvxpy result:\n', np.round(result,3))
	print(prob.value)

if __name__ == '__main__':
	task = sys.argv[1]

	# f_only(task,lam=0.1)
	# s_f(task, lam=0.00009, check_loss_only=False, split=True)
	s_f_direct(task, lam=0.00009, split=True)
	# reconstruct_err(task,fdir+'0.0014_1stage_er2_'+task+'.npy')
	# sgcrf(task)
	# cc_cvx(task=task, lam=0.01, split=True)