from scipy.io import loadmat
import numpy as np
from cc_fista import cc_fista
import time

def data_prep(Dir = '/home/sikun/Documents/data/HCP-V1/'):
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

	print(vec_s.shape)
	print(vec_f.shape)
	return vec_s, vec_f

def main():
	_, vec_f = data_prep()
	fi = cc_fista(vec_f,0.5)
	# import ipdb; ipdb.set_trace()
	start = time.time()
	invcov = fi.infer()
	print((time.time()-start)/60)

if __name__ == '__main__':
	main()