from scipy.io import loadmat
import numpy as np
from cc_fista import cc_fista
import time, yaml

# load config file
with open('config.yaml') as info:
    info_dict = yaml.load(info)

def data_prep(Dir = info_dict['data_dir']):
	sMat = loadmat(Dir+'Diffusion-q.mat')
	s = sMat['X']
	fMat = loadmat(Dir+'tfMRI-EMOTION.mat')
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
	return vec_s, vec_f

def f_only():
	_, vec = data_prep()
	fi = cc_fista(vec,0.3)
	print('Input vector shape: ', vec.shape)
	# import ipdb; ipdb.set_trace()
	start = time.time()
	invcov = fi.infer()
	print((time.time()-start)/60)
	print(np.count_nonzero(invcov))
	import ipdb; ipdb.set_trace()

def s_f():
	vec_s, vec_f = data_prep()
	vec = np.concatenate((vec_f,vec_s), axis=1)
	print('Input vector shape: ', vec.shape)
	fi = cc_fista(vec, 0.3, s_f=True)
	start = time.time()
	omega = fi.infer_s_f()
	print((time.time()-start)/60)
	import ipdb; ipdb.set_trace()

if __name__ == '__main__':
	# f_only()
	s_f()