import numpy as np
import sys
from scipy.stats.stats import pearsonr
import scipy.linalg as LA

task = sys.argv[1]
fold = int(sys.argv[2])
result_dir = sys.argv[3] #'results2'

'''helper functions'''
from scipy.linalg import cho_factor, cho_solve, LinAlgError
def check_pd(A, lower=True):
    """
    Checks if A is PD.
    If so returns True and Cholesky decomposition,
    otherwise returns False and None
    """
    try:
        return True, np.tril(cho_factor(A, lower=lower)[0])
    except LinAlgError as err:
        if 'not positive definite' in str(err):
            return False, None

def chol_inv(B, lower=True):
    """
    Returns the inverse of matrix A, where A = B*B.T,
    ie B is the Cholesky decomposition of A.
    Solves Ax = I
    given B is the cholesky factorization of A.
    """
    return cho_solve((B, lower), np.eye(B.shape[0]))

def inv(A):
    """
    Inversion of a SPD matrix using Cholesky decomposition.
    """
    return chol_inv(check_pd(A)[1])

'''data loading'''
def get_theta(task, suffix):
	# Get Theta from file
	filepath = result_dir+'/Thetafile_'+task+'_'+suffix
	with open(filepath) as fp:
		'''Get the first line, which contains result information'''
		line = fp.readline()
		info = [int(s) for s in line.strip().split(' ')]
		Theta = np.zeros((info[0],info[1]))
		nz_info = info[2]
		print('Theta non-zero entry number: ', nz_info)
		print('Theta shape: ', Theta.shape)
		while line:
			line = fp.readline()
			if line:
				line_info = [float(s) for s in line.strip().split(' ')]
				Theta[int(line_info[0]-1),int(line_info[1]-1)] = line_info[2]
	# print(np.count_nonzero(Theta))
	return Theta

def get_lam(task, suffix):
	# Get Lambda from file
	filepath = result_dir+'/Lambdafile_'+task+'_'+suffix
	with open(filepath) as fp:
		'''Get the first line, which contains result information'''
		line = fp.readline()
		info = [int(s) for s in line.strip().split(' ')]
		Lambda = np.zeros((info[0],info[1]))
		nz_info = info[2]
		print('Lambda non-zero entry number: ', nz_info)
		print('Lambda shape: ', Lambda.shape)
		while line:
			line = fp.readline()
			if line:
				line_info = [float(s) for s in line.strip().split(' ')]
				Lambda[int(line_info[0]-1),int(line_info[1]-1)] = line_info[2]
	# print(np.count_nonzero(Lambda))
	return Lambda

def get_test_X(task, suffix):
	# Load test X
	X = []
	filepath = 'data/val_s_'+task+'_'+suffix
	with open(filepath) as fp:
		line = fp.readline()
		X.append([float(s) for s in line.strip().split(' ')])
		while line:
			line = fp.readline()
			if line:
				X.append([float(s) for s in line.strip().split(' ')])
	X = [np.asarray(arr) for arr in X]
	X = np.stack(X)
	# print(X.shape)
	return X

def get_test_Y(task, suffix):
	# Load test Y
	Y = []
	filepath = 'data/val_f_'+task+'_'+suffix
	with open(filepath) as fp:
		line = fp.readline()
		Y.append([float(s) for s in line.strip().split(' ')])
		while line:
			line = fp.readline()
			if line:
				Y.append([float(s) for s in line.strip().split(' ')])
	Y = [np.asarray(arr) for arr in Y]
	Y = np.stack(Y)
	# print(Y.shape)
	return Y

def eval(pred_f, vec_f):
	mpe = []
	mape = []

	# correlation coefficient and p-value
	avg_r = []
	min_pval = 1
	max_pval = 0
	for i in range(len(vec_f)):
		mpe.append((vec_f[i] - pred_f[i]).sum() / vec_f[i].sum())
		mape.append((LA.norm(vec_f[i] - pred_f[i]) ** 2) / (LA.norm(vec_f) ** 2))
		# mape.append(np.abs((vec_f[i] - pred_f[i]).sum()) / vec_f[i].sum())
		r, pval = pearsonr(pred_f[i],vec_f[i])
		avg_r.append(r)
		if pval < min_pval:
			min_pval = pval
		if pval > max_pval:
			max_pval = pval
	
	mpe = sum(mpe)/len(mpe)
	mape = sum(mape)/len(mape)
	avg_r = sum(avg_r)/len(avg_r)
	return mpe, mape, avg_r, min_pval, max_pval

def main():
	val_mpe_avg	=	[]
	val_mape_avg	=	[]
	val_r_avg 	=	[]
	val_p_min 	=	1
	val_p_max 	=	0
	for i in range(fold):
		print('cross validation:', i)
		Theta = get_theta(task, str(i))
		Lambda = get_lam(task, str(i))
		X = get_test_X(task, str(i))
		Y_gt = get_test_Y(task, str(i))
		# Do prediction of Y using Theta, Lambda, X
		Y_pred = -np.dot(np.dot(inv(Lambda), Theta.T), X.T).T
		# print(Y_pred.shape)

		# compare with gt Y
		# np.save('pred_f_'+method+suffix+'.npy',Y_pred)
		mpe, mape, avg_r, min_p, max_p = eval(Y_pred, Y_gt)
		val_mpe_avg.append(mpe)
		val_mape_avg.append(mape)
		val_r_avg.append(avg_r)
		if min_p < val_p_min:
			val_p_min = min_p
		if max_p > val_p_max:
			val_p_max = max_p
	
	print('***VAL DATA***', fold, 'fold recon error average:', 
		'MSEP:', np.mean(val_mape_avg), '+-', np.std(val_mape_avg), 
		'MPE:', np.mean(val_mpe_avg), '+-', np.std(val_mpe_avg),
		'; r average:', np.mean(val_r_avg), np.std(val_r_avg), 
		'; min/max p value:', val_p_min, val_p_max)
if __name__ == '__main__':
	main()
