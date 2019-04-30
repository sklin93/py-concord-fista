import numpy as np
import sys

if len(sys.argv) == 2:
	method = sys.argv[1]	#method used (CD or BCD)
	suffix = ''
elif len(sys.argv) == 3:
	method = sys.argv[1]
	suffix = sys.argv[2]	#e.g. '_lang_train_0.01'
# Get Theta from file
filepath = 'results/Thetafile_'+method+suffix
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
print(np.count_nonzero(Theta))

# Get Lambda from file
filepath = 'results/Lambdafile_'+method+suffix
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
print(np.count_nonzero(Lambda))

# Load X from Xfile
# TODO: flexible filename
X = []
filepath = 'data/Xfile_lang_test'
with open(filepath) as fp:
	line = fp.readline()
	X.append([float(s) for s in line.strip().split(' ')])
	while line:
		line = fp.readline()
		if line:
			X.append([float(s) for s in line.strip().split(' ')])
X = [np.asarray(arr) for arr in X]
X = np.stack(X)
print(X.shape)

# Do prediction of Y using Theta, Lambda, X
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
Y_pred = -np.dot(np.dot(inv(Lambda), Theta.T), X.T).T
print(Y_pred.shape)
np.save('pred_f_'+method+suffix+'.npy',Y_pred)