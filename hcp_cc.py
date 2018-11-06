from scipy.io import loadmat
import numpy as np
from cc_fista import cc_fista, standardize, pseudol
import time, yaml
from scipy.linalg import norm

with open('config.yaml') as info:
    info_dict = yaml.load(info)

def data_prep(upenn=False, normalize_s=False):
	if upenn:
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
		fMat = loadmat(info_dict['data_dir']+info_dict['f_file'])
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
	return vec_s, vec_f

def s_f_pMat(d): # d is p/2
	tmp_14 = np.identity(d)*2
	tmp_23 = np.ones((d,d))
	np.fill_diagonal(tmp_23, 2)
	up = np.concatenate((tmp_14,tmp_23),axis=1)
	low = np.concatenate((tmp_23,tmp_14),axis=1)
	return np.concatenate((up,low),axis=0)

def f_only(lam):
	# original version
	_, vec = data_prep(upenn=True)
	fi = cc_fista(vec,lam,steptype=1)
	print('Input vector shape: ', vec.shape)
	start = time.time()
	invcov = fi.infer()
	print((time.time()-start)/60)
	print(np.count_nonzero(invcov))
	print(np.count_nonzero(invcov.diagonal()))
	print(fi.loss())

def s_f(lam, check_loss_only=False):
	# vec_s, vec_f = data_prep(upenn=False, normalize_s=True)
	vec_s, vec_f = data_prep(upenn=True)
	vec = np.concatenate((vec_f,vec_s), axis=1)
	print('Input vector shape: ', vec.shape)
	if check_loss_only:
		check_loss(vec,np.load(str(lam)+'.npy'))
		return
	fi = cc_fista(vec, lam, s_f=True, steptype=3, const_ss=2.0)
	start = time.time()
	omega = fi.infer()
	print((time.time()-start)/60)
	np.save(str(lam)+'.npy',omega)
	print(np.count_nonzero(omega))
	d = omega.shape[0]
	print(np.count_nonzero(omega[:,:d]))
	print(np.count_nonzero(omega[:,d:]))
	print(np.count_nonzero(omega[:,:d].diagonal()))
	print(np.count_nonzero(omega[:,d:].diagonal()))
	print(fi.loss())

def s_f_direct(lam):
	vec_s, vec_f = data_prep(upenn=True)
	vec = np.concatenate((vec_f,vec_s), axis=1)
	print('Input vector shape: ', vec.shape)
	pMat = s_f_pMat(int(vec.shape[1]/2))
	fi = cc_fista(vec, lam, pMat=pMat, steptype=3, const_ss=2.0)
	start = time.time()
	omega = fi.infer()
	print((time.time()-start)/60)
	import ipdb; ipdb.set_trace()
	np.save('direct_'+str(lam)+'.npy',omega)

def check_loss(D,X):
	print(np.count_nonzero(X))
	d = omega.shape[0]
	print(np.count_nonzero(X[:,:d]))
	print(np.count_nonzero(X[:,d:]))
	print(np.count_nonzero(X[:,:d].diagonal()))
	print(np.count_nonzero(X[:,d:].diagonal()))
	S = standardize(D)
	print(pseudol(X,S@X.transpose()))

def reconstruct_err(filename):
	vec_s, vec_f = data_prep(upenn=True)
	n = vec_s.shape[0]
	omega = np.load(filename)
	d = omega.shape[0]

	vec = np.concatenate((vec_f,vec_s), axis=1)
	S = standardize(vec)
	print((omega.transpose()*(S@omega.transpose())).sum())
	import ipdb; ipdb.set_trace()
	omega = omega[:,d:]
	err = tot_err_perct = 0.0
	for k in range(n):
		cur_s = vec_s[k]
		f_gt = vec_f[k]
		f_model = np.zeros(d)
		for i in range(d):
			f_model[i] = (omega[i].dot(cur_s))
		import ipdb; ipdb.set_trace()
		cur_err = norm(f_model-f_gt)
		err_perct = cur_err/norm(f_gt)
		print(err_perct)
		tot_err_perct += err_perct
		err += cur_err
	print('average error percentage',tot_err_perct/n)
	return err

if __name__ == '__main__':
	# f_only(lam=0.1)
	# s_f(lam=0.16, check_loss_only=False)
	# s_f_direct(lam=0.12)
	print(reconstruct_err('0.08.npy'))