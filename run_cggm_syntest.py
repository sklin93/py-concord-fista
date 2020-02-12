import numpy as np
import cvxpy as cvx
import scipy.linalg as LA
import sys, os, pickle, argparse, math, time, pickle
from scipy.stats.stats import pearsonr



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



def load_model_param(filepath):

    with open(filepath) as fp:
        '''Get the first line, which contains result information'''
        line = fp.readline()
        info = [int(s) for s in line.strip().split(' ')]
        A = np.zeros((info[0],info[1]))
        nz_info = info[2]
        while line:
            line = fp.readline()
            if line:
                line_info = [float(s) for s in line.strip().split(' ')]
                A[int(line_info[0]-1),int(line_info[1]-1)] = line_info[2]
    return A


def eval_cggm(pred_f, vec_f):

    # correlation coefficient and p-value
	mpe = [];      mape = [];       avg_r = [];
	min_pval = 1;  max_pval = 0;
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


def eval_fpr(A_ori, A_hat):

    # FPR
    num_ori = np.count_nonzero(A_ori)
    num_hat = np.count_nonzero(A_hat) 

    num_TP = np.count_nonzero(np.logical_and.reduce([A_ori, A_hat]))
    num_FN = num_ori - num_TP
    num_FP = num_hat - num_TP
    num_TN = A_ori.size - num_ori

    FPR = num_FP / (num_FP + num_TN)
    TPR = num_TP / (num_TP + num_FN)

    return FPR, TPR


def run_cggm(args):

    with open(args.synthetic_data, 'rb') as pfile:
        data = pickle.load(pfile)

    prefix = os.path.splitext(os.path.basename(args.synthetic_data))[0]
    wkdir       = "mega-cggm/demo/"+prefix
    file_input  = wkdir+"/X_train.txt"
    file_output = wkdir+"/Y_train.txt"
    file_lambda = wkdir+"/lambda.txt"
    file_theta  = wkdir+"/theta.txt"
    file_stats  = wkdir+"/stats.txt"

    os.system("mkdir "+wkdir)

    # generate text files for X and Y
    np.savetxt(file_input,  data.X, delimiter=" ")
    np.savetxt(file_output, data.Y, delimiter=" ")

    # run the program
    if args.run_cggm:
        command_str = "mega-cggm/AltNewtonCD/cggmfast_run -y "+str(args.lambda1)+" -x "+str(args.lambda2)+" -v 0 "+str(data.n)+" "+str(data.p)+" "+str(data.n)+" "+str(data.q)+" "+file_output+" "+file_input+" "+file_lambda+" "+file_theta+" "+file_stats
        print(command_str)
        os.system(command_str)
    
    # get results
    theta = load_model_param(file_theta)
    print('[Theta] non-zeros: ', np.count_nonzero(theta))
    lambd = load_model_param(file_lambda)
    print('[Lambda] non-zeros: ', np.count_nonzero(lambd))

    # Do prediction of Y using Theta, Lambda given X_test
    Y_pred = -np.dot(np.dot(inv(lambd), theta.T), data.X_test.T).T
    mpe, mape, avg_r, min_p, max_p = eval_cggm(Y_pred, data.Y_test)
    print('***VAL DATA***\nMPE:{0:.3e}, MSE:{1:.3e}\nCorr:{2:.3e}, Min_pval:{3:.3e}, Max_pval:{4:.3e}'.format(mpe, mape, avg_r, min_p, max_p))

    B_hat = -np.dot(theta, inv(lambd))
    FPR, TPR = eval_fpr(data.B, B_hat)
    print('[B_hat] FPR: {0:.3f}, TPR: {1:.3f}, non-zeros:{2:d}'.format(FPR, TPR, np.count_nonzero(B_hat)))

    Omg_hat = lambd
    FPR, TPR = eval_fpr(data.Omg, Omg_hat)
    print('[Omg_hat] FPR: {0:.3f}, TPR: {1:.3f}, non-zeros:{2:d}'.format(FPR, TPR, np.count_nonzero(Omg_hat)))

    return 


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for CGGM.')

    parser.add_argument('--synthetic_data', type=str, default='data-utility/syndata_n50p20q20.pkl',
                        help='File path to the synthetic dataset')
    parser.add_argument('--lambda1', type=float, default=1,
                        help='lambda1: penalty parameter for l_1')
    parser.add_argument('--lambda2', type=float, default=1,
                        help='lambda2: penalty parameter for l_2')
    parser.add_argument('--run_cggm', default=False, action='store_true',
                        help='Flag to call CGGM, otherwise simply evaluate')

    args = parser.parse_args()
    run_cggm(args)

