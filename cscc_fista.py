import csv
import numpy as np
import cvxpy as cvx
import sys, os, pickle, argparse

from math import sqrt
from scipy import sparse
from scipy.stats import ortho_group
from scipy.linalg import norm, inv
from pprint import pprint

import matplotlib.pyplot as plt





class cscc_fista(object):
    """ Convex set constrained CONCORD with a two-stage FISTA solver """

    def __init__(self, D, num_var, sample_cov=False, record=True, pMat=None, 
        p_gamma=1.0, p_lambda=1.0, verbose=True, MAX_ITR=300, TOL=1e-5, 
        p_tau=1, c_outer=0.9, alpha_out=1.0, step_type_out=1, const_ss_out=0.1,
        verbose_inn=False, MAX_ITR_inn=100, TOL_inn=1e-7, p_kappa=0.5, 
        c_inner=0.9, alpha_inn=1.0, step_type_inn=3, verbose_inn_details=False,
        no_constraints=False, inner_cvx_solver=False):

        super(cscc_fista, self).__init__()
        self.record      = record
        self.no_constraints = no_constraints
        self.inner_cvx_solver = inner_cvx_solver

        # inout parameters of primal problem
        self.S           = D.copy() if sample_cov else self.get_sample_cov(D)
        self.pMat        = pMat.copy()
        self.p_gamma     = p_gamma
        self.p_lambda    = p_lambda
        self.num_var     = num_var

        # outer stage parameters
        self.verbose     = verbose
        self.MAX_ITR     = MAX_ITR
        self.TOL         = TOL
        self.p_tau       = p_tau
        self.c_outer     = c_outer
        self.alpha_out   = alpha_out
        self.step_type_out  = step_type_out
        self.const_ss_out = const_ss_out

        # inner stage parameters
        self.verbose_inn = verbose_inn
        self.MAX_ITR_inn = MAX_ITR_inn
        self.TOL_inn     = TOL_inn
        self.p_kappa     = p_kappa
        self.c_inner     = c_inner
        self.alpha_inn   = alpha_inn
        self.step_type_inn  = step_type_inn
        self.verbose_inn_details = verbose_inn_details

        # solution initialization
        # make sure initial B_init has all zero diagonals
        self.Omg_init   = np.identity(num_var)
        self.B_init     = np.zeros(self.Omg_init.shape)

        # meta variables
        self.A           = np.zeros(self.Omg_init.shape)
        self.A_X         = np.zeros(self.Omg_init.shape)

        """
        Notes:
            if step_type_out is 3, use self.p_tau as constant step length in the outer stage; 
            if step_type_inn is 3, use self.p_kappa as constant step length in the inner stage.
        """

    def get_sample_cov(self, D):
        """ comupte sample covariance S from data matrix D.
        S is invariant to mean shift on D.
        Input:
            D: n(sample)-by-d(dimension) matrix,
        """
        num_sample = D.shape[0]
        Y = D - np.tile(D.mean(axis=0), (num_sample,1))
        # should we use biased S-estimator?
        S = Y.transpose() @ Y / (num_sample - 1)
        return S

    def likelihood_convset(self, Omg, SOmg):
        """ OUTER stage objective 
        h(Omega) = -2*log(Omega) + tr(S @ Omega^2)
        Input:
            Omg:  Omega (d-by-d)
            SOmg: S * Omega (d-by-d)
        """
        # for the gradient and likelihood, should we use log(abs(det))?
        # Accelerate via equality: tr(A.transpose()@A) = (A*A).sum()
        return -2 * np.log(Omg.diagonal()).sum() + (Omg.transpose()*SOmg).sum()

    def likelihood_linfty(self, W):
        """ INNER stage objective 
        W = A_x - gamma * lambda * B_x
        g(B_x) = ||W||^2 - ||P_perp_to_c (W)||^2
        Input:
            W: d-by-d
        """
        # make sure that A_X and B both have empty diagonals
        W_c = W * self.pMat  # W projected onto C
        return norm(W)**2 - norm(W - W_c)**2

    def update_convset(self, Th, G, tau):
        """ OUTER stage, update Omg_t under convex set constraint """
        # gradient descent step
        self.A   = Th - tau * G
        self.A_X = self.A.copy()
        np.fill_diagonal(self.A_X, 0)

        if not self.no_constraints:
            # Use inner stage to compute current optimal B.
            # You can use CVX library to check the correctness of
            # our FISTA implementation of inner problem solver.
            if self.inner_cvx_solver:
                B = self.solver_linfty_cvx()
            else:
                B = self.solver_linfty()
            np.fill_diagonal(B, 0)

            # proximal operator of convex set constraint
            # pMat(i,j) = 0 if (e_i, e_j) is prohibited in the solution
            W   = self.A_X - self.p_gamma * self.p_lambda * B
            # add diagonal entries from A
            Omg = W * self.pMat + np.diag(np.diag(self.A))
        
        else:
            # print("No constraint is applied.")
            LambdaMat = self.p_lambda * np.ones((self.num_var, self.num_var))
            np.fill_diagonal(LambdaMat, 0)
            Omg = np.sign(self.A) * np.maximum(abs(self.A)-tau*LambdaMat, 0.0)

        return Omg

    def solver_linfty_cvx(self):
        """cvx version of inner stage solver,
        specifically for edge-forbidden constraints"""
    
        A_x  = self.A_X
        dim  = A_x.shape[0]
        B    = cvx.Variable((dim, dim))
        loss = 0
        for i in range(dim):
            for j in range(dim):
                if i == j: 
                    continue
                if self.pMat[i,j] == 1:
                    loss += cvx.norm(A_x[i,j]-self.p_gamma*self.p_lambda*B[i,j]) ** 2

        obj = cvx.Minimize(loss)
        constraints = [-1 <= B, B <= 1]

        prob = cvx.Problem(obj, constraints)
        prob.solve(verbose=False)
        # print("Is this problem DGP?", prob.is_dgp())

        if self.verbose_inn:
            print("\n- - - solving inner problem with CVXPY - - - ")
            # print("status:", prob.status)
            print("optimal value:", prob.value)
            print("solution B_x: "); print(B.value)

        return B.value

    def update_linfty(self, Th_, G_, kappa):
        """ update B_t' under l_infty norm constraint """

        """ proximal operator of l_infty norm constraint
        Note: Such update strategy will guarantee all zeros on diagonal of B,
        if the initial Th_ has all zero diagonals. """
        A_ = Th_ - kappa * G_
        if self.verbose_inn_details:
            print("inner stage [B_X before proximal operation]")
            pprint(A_)
        B  = np.sign(A_) * np.minimum(abs(A_), np.ones(A_.shape))
        if self.verbose_inn_details:
            print("inner stage [B_X after proximal operation]")
            pprint(B)
        return B

    def solver_convset(self):
        """ outer stage optimization via FISTA """

        tau_n  = self.p_tau
        alpha  = self.alpha_out
        Lambda = self.p_lambda * np.ones(self.Omg_init.shape)
        if self.step_type_out == 3:
            tau_n = self.const_ss_out

        # Omega initial likelihood
        Omg  = self.Omg_init.copy()
        SOmg = self.S @ Omg
        h    = self.likelihood_convset(Omg, SOmg)

        # Theta & initial gradient
        Th   = self.Omg_init.copy()
        ThS  = Th @ self.S  # Theta*S or S*Theta
        G = - np.diag(1.0/Th.diagonal()) + 0.5*(ThS + ThS.transpose())

        # initial A
        self.A   = Th - self.p_tau * G
        self.A_X = self.A.copy()

        plt.ion() ## Note this correction
        plt.figure()
        plot_x=list()
        plot_y=list()
        # plt.axis([0, self.MAX_ITR, 0,1])

        # looping for optimization steps
        loop  = True
        itr   = itr_back = iter_diag = 0
        h_n   = Q_n      = f         = 0.0
        while loop:
            itr_diag = 0
            itr_back = 0
            tau      = tau_n

            if self.verbose: 
                print("\n\n\n\n = = = iteration "+str(itr)+" = = = ")

            # constant step length
            if self.step_type_out == 3:
                Omg_n   = self.update_convset(Th, G, tau)
                # print("Omega_n="); print(Omg_n)
                SOmg_n  = self.S @ Omg_n
                h_n     = self.likelihood_convset(Omg_n, SOmg_n)
            # looping for adaptive step length as backtacking line search
            else:
                while True:
                    if itr_diag !=0 or itr_back != 0:
                        tau = tau * self.c_outer
                    Omg_n = self.update_convset(Th, G, tau)
                    # if solution has zeros on diagonal, continue
                    if Omg_n.diagonal().min() < 1e-8 and itr_diag < 50:
                        itr_diag += 1
                        continue

                    # check backtracking condition
                    Step    = Omg_n - Th
                    Q_n     = h + (Step*G).sum() + (1/(2*tau))*(norm(Step)**2)
                    SOmg_n  = self.S @ Omg_n
                    h_n     = self.likelihood_convset(Omg_n, SOmg_n)
                    if h_n > Q_n: # sufficient descent condition
                        itr_back += 1
                    else:
                        break
                # end of (while True)
            # end of else
            if self.verbose:
                Omg_x = Omg_n.copy()
                np.fill_diagonal(Omg_x, 0)
                f_n = h_n+self.p_lambda*np.linalg.norm(Omg_x,1)
                print("\n- - - OUTER problem solution UPDATED - - -\n" + \
                    "1st term: "+"{:.6f}".format(-2 * np.log(Omg_n.diagonal()).sum()) + \
                    " | 2nd term: "+"{:.6f}".format((Omg_n.transpose()*SOmg_n).sum()) + \
                    " | objective: "+"{:.6f}".format(f_n))

            # FISTA momentum update step
            alpha_n = (1 + sqrt(1 + 4*(alpha**2)))/2
            Th  = Omg_n + ((alpha-1)/alpha_n) * (Omg_n - Omg)
            # update meta variable
            ThS = Th @ self.S
            # update gradient
            G_n = - np.diag(1.0/Th.diagonal()) + 0.5*(ThS + ThS.transpose())
            # update tau for next opt iteration
            if self.step_type_out == 0:
                tau_n = 1
            elif self.step_type_out == 1:
                tau_n = tau
            elif self.step_type_out == 2:
                tau_n = (Step * Step).sum() / (Step * (G_n - G)).sum()
                tau_n = tau if tau_n < 0.0 else tau_n
                # taun = (Step.transpose()@Step).trace() \
                #              / (Step.transpose()@(Gn-G)).trace()
                # using *.sum() is much faster

            # update for next opt iteration
            alpha   = alpha_n
            Omg     = Omg_n
            h       = h_n
            G       = G_n
            itr     += 1

            # won't be used, just for printing
            # f       = h + (abs(Omg_n)).sum()

            # compute subgradient error:
            # 1. As Omg_n has been located in the constrained convex set, it is
            # straightforward to compute subgradient at Omg_n
            # 2. The following code actually computes subgradient at updated 
            # momentum position Theta_n, except the case that Omg_n(i,j) = 0
            # which is calculated with l1 subgradient definition at zero.
            # 3. Why does we use the sign of Omg_n rather than Th? """
            tmp       = G_n + np.sign(Omg_n) * Lambda # sign(Omg_n) or sign(Th)
            subg      = np.sign(G_n) * np.maximum(abs(G_n) - Lambda, 0.0)
            subg[Omg_n != 0] = tmp[Omg_n != 0]
            cur_err   = norm(subg) / norm(Omg_n)

            if self.verbose: 
                # print("updated Omega:"); print(Omg)
                # print("updated Theta:"); print(Th)
                print("error: "+"{:.2f}".format(cur_err)+\
                    ", subg norm:"+"{:.2f}".format(norm(subg))+\
                    "\nh function value:"+"{:.6f}".format(h)+\
                    "\nh function comparable value:"+"{:.6f}".format(h/2))
                print("Inferred Omega:")
                print(Omg_n)
                print('nonzero entry count: ', np.count_nonzero(Omg_n))
                if np.isnan(h): sys.exit()
            
            plot_x.append(itr) 
            plot_y.append(f_n) 
            plt.plot(plot_x, plot_y, 'bo-')
            plt.show(block=False)
            plt.pause(0.0001)

            # check termination condition:
            loop = itr < self.MAX_ITR and cur_err > self.TOL

            if self.record:
                with open('itrloss_' + str(self.p_lambda)+'.csv', 'a') as f:
                    fwriter = csv.writer(f)
                    fwriter.writerow([itr] + [cur_err])
        # end of (while loop)

        plt.show(block=True)
        self.result = Omg_n.copy()
        return Omg_n
    # end of solver_convset

    def solver_linfty(self):
        """ inner stage optimization via FISTA """
        """ The only input parameter of the problem is given as self.A_X """

        kappa_n = self.p_kappa
        alpha   = self.alpha_inn

        # B and initial likelihood
        B   = self.B_init.copy()
        W   = self.A_X - self.p_gamma * self.p_lambda * B
        g   = self.likelihood_linfty(W)

        # Theta_ initialization and get initial gradient
        Th_ = self.B_init.copy()
        G_  = - 2 * self.p_lambda * self.p_gamma * self.pMat * W

        # looping for optimization steps
        loop  = True
        itr   = itr_back = 0
        g_n   = R_n      = 0.0
        if self.verbose_inn: 
            print("\n - - - solving inner problem with FISTA - - -")
        while loop:
            itr_back = 0
            kappa    = kappa_n

            if self.verbose_inn_details: 
                print(">>> inner loop [itr " + str(itr) + "]:")

            # constant step length
            if self.step_type_inn == 3:
                B_n = self.update_linfty(Th_, G_, kappa)
                W_n = self.A_X - self.p_gamma * self.p_lambda * B_n
                g_n = self.likelihood_linfty(W_n)
            # looping for adaptive step length as backtacking line search
            else:
                while True:
                    if itr_back != 0:
                        kappa = kappa * self.c_inner
                    B_n = self.update_linfty(Th_, G_, kappa)

                    # check backtracking condition
                    Step = B_n - Th_
                    R_n  = g + (Step*G_).sum() + (1/(2*kappa)) * (norm(Step)**2)
                    W_n  = self.A_X - self.p_gamma * self.p_lambda * B_n
                    g_n  = self.likelihood_linfty(W_n)
                    if g_n > R_n: # sufficient descent condition
                        itr_back += 1
                    else:
                        break
                # end of (while True)
            # end of else

            # FISTA momentum update step
            alpha_n = (1 + sqrt(1 + 4*(alpha**2)))/2
            Th_  = B_n + ((alpha-1)/alpha_n) * (B_n - B)
            # update gradient
            W_n  = self.A_X - self.p_gamma * self.p_lambda * B_n
            G_n_ = - 2 * self.p_lambda * self.p_gamma * self.pMat * W_n
            # update tau for next opt iteration
            if self.step_type_inn == 0:
                kappa_n = 1
            elif self.step_type_inn == 1:
                kappa_n = kappa
            elif self.step_type_inn == 2:
                kappa_n = (Step * Step).sum() / (Step * (G_n_ - G_)).sum()
                kappa_n = kappa if kappa_n < 0.0 else kappa_n
                """ kappa_n = (Step.transpose()@Step).trace() \
                             / (Step.transpose()@(G_n_-G_)).trace()
                    using *.sum() is much faster """

            """ compute gradient error
            As B_n has been located in the bounding box w.r.t. l_infty <= 1, 
            we can directly use the gradient of g which is quaratic."""
            cur_err   = norm(G_n_) / norm(B_n)
            # check termination condition:
            # loop = itr < self.MAX_ITR_inn and cur_err > self.TOL_inn # gradient condition
            loop = itr < self.MAX_ITR_inn and abs(g_n - g)/abs(g) > self.TOL_inn # g_val condition

            # update for next opt iteration
            alpha   = alpha_n
            B       = B_n
            g       = g_n
            G_      = G_n_
            itr     += 1

            # print
            if self.verbose_inn_details: 
                print(" >>> cur_err: " + str(cur_err) + ", g(B_n) = " + str(g))

        # end of (while loop)
        if self.verbose_inn:
            print("optimal value of g:", g)
            print("solution B_x: "); print(B)

        return B_n
    # end of solver_convset


def set_mat_from_triu(vec, num_var, nnz_index):
    """ ARCHIVED
        Build a symmetric matrix [M] from
        a given vector [vec] which contains all upper triangular entries
        and a corresponding non-zero index vector [nnz_index] """

    num_edge = int(num_var*(num_var-1)/2)
    vec[np.setdiff1d(range(num_edge), nnz_index)] = 0
    M = np.zeros((num_var, num_var))
    M[np.triu_indices(M.shape[0], 1)] = vec
    M = M + M.transpose() + np.identity(num_var)
    return M

def create_sparse_mat(num_var):
    """ ARCHIVED """
    num_nnz   = 3
    num_edge  = int(num_var*(num_var-1)/2)
    nnz_index = np.random.choice(range(num_edge), num_nnz, replace=False)
    nnz_index.sort()

    # generate inverse covariance matrix

    cov_vec   = np.random.rand(num_edge) * 0.3 + 0.7
    Omg = set_mat_from_triu(cov_vec, num_var, nnz_index)

    # generate convex set mask
    mask_vec = np.ones(num_edge)
    pMat = set_mat_from_triu(mask_vec, num_var, nnz_index)

    return [Omg, pMat]

def generate_synthetic(syndata_file):

    num_var   = 7    # number of variables
    num_smp   = 200  # number of samples
    pct_nnz   = 0.2  # percentage of non-zero entries in L matrix
    base_nnz  = 0.7  # base value of non-zero entries in L matrix

    # randomly select a certain number of edges
    # as non-zeros in partial correlation graph

    # create sparse symmetric positive definite matrix:
    # Every positive-definite matrix has a Cholesky decomposition that takes the form LL' where L is lower triangular, 
    # so sample L and compute a positive-definite matrix from it. If L is sparse then LL' is also sparse. 
    # Make sure L is less sparse then what you want your final matrix to be.
    Spr = sparse.random(num_var, num_var, density=pct_nnz).A
    Spr[Spr != 0] = Spr[Spr != 0] * (1 - base_nnz) + base_nnz
    print("triu-nonzeros of Spr: " + str(np.count_nonzero(Spr)))
    Chol = np.tril(Spr)
    np.fill_diagonal(Chol, 1)
    Omg  = Chol @ Chol.transpose()
    print("triu-nonzeros of Omg: " + str((np.count_nonzero(Omg)-num_var)/2.0))

    # create convex set mask
    pMat = Omg.copy()
    pMat[pMat != 0] = 1
    # num_nnz = np.count_nonzero(pMat)

    # generate samples from target distribution
    Sig = inv(Omg)
    D   = np.random.multivariate_normal(np.zeros(num_var), Sig, num_smp)
    pprint(Omg)

    # save generated model and samples
    with open(syndata_file, 'wb') as p:
        pickle.dump((Omg, Sig, D, pMat, num_var, num_smp), p)

    return


def test_synthetic(syndata_file, args):

    print("Loading synthetic dataset ... \n")
    with open(syndata_file, 'rb') as p:
        (Omg, Sig, D, pMat, num_var, num_smp) = pickle.load(p)
    print("Loaded ... Groundtruth Omega:")
    print(Omg)
    
    # partial correlation graph estimation
    problem  = cscc_fista(D, num_var=num_var, pMat=pMat, 
                    MAX_ITR=args.MAX_ITR,
                    step_type_out = args.step_type_out, const_ss_out = args.const_ss_out, 
                    p_gamma=args.p_gamma, p_lambda=args.p_lambda, p_tau=args.p_tau, 
                    TOL=args.TOL, TOL_inn=args.TOL_inn,
                    verbose=args.outer_verbose, verbose_inn=args.inner_verbose,
                    no_constraints=args.no_constraints, inner_cvx_solver=args.inner_cvx_solver)
    Omg_hat  = problem.solver_convset()

    # output results
    print("\n\n= = = Finished = = =\nGroundtruth Omega:")
    print(Omg)
    print('nonzero entry count: ', np.count_nonzero(Omg))
    print("Inferred Omega:")
    print(Omg_hat)
    print('nonzero entry count: ', np.count_nonzero(Omg_hat))
    return



def main(args):

    np.set_printoptions(formatter={'float': '{: 0.2f}'.format})

    if args.generate_synthetic:

        syndata_file = args.synthetic_dir
        if not os.path.isfile(syndata_file):
            print("Generating synthetic dataset ... \n")
            generate_synthetic(syndata_file)

    if args.demo:
        # syndata_file = 'data-utility/syn.pkl'
        syndata_file = args.synthetic_dir
        test_synthetic(syndata_file, args)



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Arguments for Constrained CONCORD.')
    
    parser.add_argument('--generate_synthetic', default=False, action='store_true',
                        help='Whether to generate a new synthetic dataset')
    parser.add_argument('--synthetic_dir', type=str, default='data-utility/new_syn.pkl',
                        help='File path to the new synthetic dataset')
    parser.add_argument('--input_dim', type=int, default=7,
                        help='Number of dimensions for input variables')

    # Parameters of algorithm
    parser.add_argument('--MAX_ITR', type=int, default=20,
                        help='Maximum iteration of outer loop')
    parser.add_argument('--MAX_ITR_inn', type=int, default=50,
                        help='Maximum iteration of inner loop')
    parser.add_argument('--step_type_out', type=int, default=3,
                        help='Type of step length setting')
    parser.add_argument('--const_ss_out', type=float, default=0.1,
                        help='Constant step length')
    parser.add_argument('--p_gamma', type=float, default=0.1,
                        help='gamma: penalty parameter in proximal operator')
    parser.add_argument('--p_lambda', type=float, default=0.2,
                        help='lambda: penalty parameter for l_1')
    parser.add_argument('--p_tau', type=float, default=0.2,
                        help='tau: step length')
    parser.add_argument('--TOL', type=float, default=1e-3,
                        help='Tolerance in outer loop')
    parser.add_argument('--TOL_inn', type=float, default=1e-2,
                        help='Tolerance in inner loop')

    # Verbose
    parser.add_argument('--inner_verbose', default=False, action='store_true',
                        help='Whether to display optimization updates of inner loop')
    parser.add_argument('--outer_verbose', default=True, action='store_true',
                        help='Whether to display optimization updates of outer loop')

    # Options of algorithm, use to compare with standard setting
    parser.add_argument('--inner_cvx_solver', default=False, action='store_true',
                        help='Use cvx solver in inner loop.')
    parser.add_argument('--no_constraints', default=False, action='store_true', 
                        help='Solve the problem with no constraints.')
    parser.add_argument('--demo', default=False, action='store_true', help='Show demo')    

    args = parser.parse_args()
    main(args)