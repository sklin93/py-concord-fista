import csv
import numpy as np
import cvxopt as cvx

from math import sqrt
from scipy import sparse
from scipy.linalg import norm, inv
from pprint import pprint


class csc_concord_fista(object):
    """ Convex set constrained CONCORD with a two-stage FISTA solver """

    def __init__(self, D, num_var, sample_cov=False, record=True, pMat=None, p_gamma=1.0, p_lambda=1.0,
        verbose=True, MAX_ITR=300, TOL=1e-5, p_tau=0.5, c_outer=0.9, alpha_out=1.0, step_type_out=3,
        verbose_inn=True, MAX_ITR_inn=300, TOL_inn=1e-5, p_kappa=0.5, c_inner=0.9, alpha_inn=1.0, step_type_inn=3):

        super(csc_concord_fista, self).__init__()
        self.record      = record

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

        # inner stage parameters
        self.verbose_inn = verbose_inn
        self.MAX_ITR_inn = MAX_ITR_inn
        self.TOL_inn     = TOL_inn
        self.p_kappa     = p_kappa
        self.c_inner     = c_inner
        self.alpha_inn   = alpha_inn
        self.step_type_inn  = step_type_inn

        # solution initialization
        # make sure initial B_init has all zero diagonals
        self.Omg_init   = np.identity(num_var)
        self.B_init     = np.zeros(self.Omg_init.shape)

        # meta variables
        self.A           = np.zeros(self.Omg_init.shape)
        self.A_X         = np.zeros(self.Omg_init.shape)

        """
        Notes:
            if step_type_out is 3, then we will use self.p_tau as constant step length
            in the outer stage; if step_type_inn is 3, then we'll use self.p_kappa as 
            constant step length in the inner stage.
        """

    def get_sample_cov(self, D):
        """ comupte sample covariance S from data matrix D """

        num_sample = D.shape[0]
        Y = D - np.tile(D.mean(axis=0), (num_sample,1))
        # should we use biased S-estimator?
        S = Y.transpose() @ Y / (num_sample - 1)
        return S

    def likelihood_convset(self, Omg, SOmg):
        """ likelihood of outer stage problem """

        # for the gradient and likelihood, should we use log(abs(det))?
        return -2 * np.log(Omg.diagonal()).sum() + (Omg.transpose()*SOmg).sum()

    def likelihood_linfty(self, W):
        """ likelihood of inner stage problem """

        # make sure that A_X and B both have empty diagonals
        W_c = W * self.pMat
        return norm(W)**2 - norm(W - W_c)**2


    def update_convset(self, Th, G, tau):
        """ update Omg_t under convex set constraint """

        # gradient descent step
        self.A   = Th - tau * G
        self.A_X = self.A.copy()
        np.fill_diagonal(self.A_X, 0)

        # use inner stage to compute current optimal B
        B   = self.solver_linfty()

        # proximal operator of convex set constraint
        # pMat(i,j) = 0 if (e_i, e_j) is prohibited in the solution
        W   = self.A_X - self.p_gamma * self.p_lambda * B
        # add diagonal entries from A
        Omg = W * self.pMat + np.diag(np.diag(self.A))

        return Omg

    def solver_linfty_cvx(self):
        """cvx version of inner stage solver"""
        pass

    def update_linfty(self, Th_, G_, kappa):
        """ update B_t' under l_infty norm constraint """

        """ proximal operator of l_infty norm constraint
        Note: Such update strategy will guarantee all zeros on diagonal of B,
        if the initial Th_ has all zero diagonals. """
        A_ = Th_ - kappa * G_
        if self.verbose_inn:
            print("inner stage [B_X before proximal operation]")
            pprint(A_)
        B  = np.sign(A_) * np.minimum(abs(A_), np.ones(A_.shape))
        if self.verbose_inn:
            print("inner stage [B_X after proximal operation]")
            pprint(B)
        return B

    def solver_convset(self):
        """ outer stage optimization via FISTA """

        tau_n  = self.p_tau
        alpha  = self.alpha_out
        Lambda = self.p_lambda * np.ones(self.Omg_init.shape)

        # Omega initial likelihood
        Omg  = self.Omg_init.copy()
        SOmg = self.S @ Omg
        h    = self.likelihood_convset(Omg, SOmg)

        # Theta & initial gradient
        Th   = self.Omg_init.copy()
        ThS  = Th @ self.S  # Theta*S or S*Theta
        G = -2 * np.diag(1.0/Th.diagonal()) + 2 * ThS

        # initial A
        self.A   = Th - self.p_tau * G
        self.A_X = self.A.copy()

        # looping for optimization steps
        loop  = True
        itr   = itr_back = iter_diag = 0
        h_n   = Q_n      = f         = 0.0
        while loop:
            itr_diag = 0
            itr_back = 0
            tau      = tau_n

            # constant step length
            if self.step_type_out == 3:
                Omg_n = self.update_convset(Th, G, tau)
            # looping for adaptive step length as backtacking line search
            else:
                while True:
                    if itr_diag !=0 or itr_back != 0:
                        tau = tau * self.c_out
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

            # FISTA momentum update step
            alpha_n = (1 + sqrt(1 + 4*(alpha**2)))/2
            Th  = Omg_n + ((alpha-1)/alpha_n) * (Omg_n - Omg)
            # update meta variable
            ThS = Th @ self.S
            # update gradient
            G_n = -2 * np.diag(1.0/Th.diagonal()) + 2 * ThS
            # update tau for next opt iteration
            if self.step_type_out == 0:
                tau_n = 1
            elif self.step_type_out == 1:
                tau_n = tau
            elif self.step_type_out == 2:
                tau_n = (Step * Step).sum() / (Step * (G_n - G)).sum()
                tau_n = tau if tau_n < 0.0 else tau_n
                """ taun = (Step.transpose()@Step).trace() \
                             / (Step.transpose()@(Gn-G)).trace()
                using *.sum() is much faster """

            # update for next opt iteration
            alpha   = alpha_n
            Omg     = Omg_n
            h       = h_n
            G       = G_n
            itr     += 1

            # won't be used, just for printing
            # f       = h + (abs(Omg_n)).sum()

            """ compute subgradient error
            1. As Omg_n has been located in the constrained convex set, it is
            straightforward to compute subgradient at Omg_n
            2. The following code actually computes subgradient at updated 
            momentum position Theta_n, except the case that Omg_n(i,j) = 0
            which is calculated with l1 subgradient definition at zero.
            3. Why does we use the sign of Omg_n rather than Th? """
            tmp       = G_n + np.sign(Omg_n) * Lambda # sign(Omg_n) or sign(Th)
            subg      = np.sign(G_n) * np.maximum(abs(G_n) - Lambda, 0.0)
            subg[Omg_n != 0] = tmp[Omg_n != 0]
            cur_err   = norm(subg) / norm(Omg_n)

            # check termination condition:
            loop = itr < self.MAX_ITR and cur_err > self.TOL

            if self.record:
                with open('itrloss_' + str(self.p_lambda)+'.csv', 'a') as f:
                    fwriter = csv.writer(f)
                    fwriter.writerow([itr] + [cur_err])

        # end of (while loop)

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
        while loop:
            itr_back = 0
            kappa    = kappa_n

            if self.verbose_inn: print("\n - - - inner loop [itr " + str(itr) + "]:")

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
            loop = itr < self.MAX_ITR_inn and cur_err > self.TOL_inn # gradient condition
            loop = itr < self.MAX_ITR_inn and abs(g_n - g)/abs(g) > self.TOL_inn # g_val condition

            # update for next opt iteration
            alpha   = alpha_n
            B       = B_n
            g       = g_n
            G_      = G_n_
            itr     += 1

            # print
            print(" - - - cur_err: " + str(cur_err) + ", g(B_n) = " + str(g))

        # end of (while loop)

        return B_n
    # end of solver_convset


def set_mat_from_triu(vec, num_var, nnz_index):
    """ Build a symmetric matrix [M] from
        a given vector [vec] which contains all upper triangular entries
        and a corresponding non-zero index vector [nnz_index] """

    num_edge = int(num_var*(num_var-1)/2)
    vec[np.setdiff1d(range(num_edge), nnz_index)] = 0
    M = np.zeros((num_var, num_var))
    M[np.triu_indices(M.shape[0], 1)] = vec
    M = M + M.transpose() + np.identity(num_var)
    return M

def create_sparse_mat(num_var):

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


def test_inner_stage():

    """ test correctness of inner-stage FISTA solver"""

    pass


def test_synthetic():

    num_var   = 5
    num_smp   = 200
    pct_nnz   = 0.3
    base_nnz  = 0.7

    # randomly select a certain number of edges
    # as non-zeros in partial correlation graph

    # create sparse symmetric positive definite matrix
    Spr = sparse.random(num_var, num_var, density=pct_nnz).A
    Spr[Spr != 0] = Spr[Spr != 0] * (1 - base_nnz) + base_nnz
    print("triu-nonzeros of Spr: " + str(np.count_nonzero(Spr)))
    Omg = Spr @ Spr.transpose()
    np.fill_diagonal(Omg, 0)
    Omg = Omg + np.identity(num_var)
    print("triu-nonzeros of Omg: " + str((np.count_nonzero(Omg)-num_var)/2.0))

    # create convex set mask
    pMat = Omg.copy()
    pMat[pMat != 0] = 1
    # num_nnz = np.count_nonzero(pMat)

    # generate samples from target distribution
    Sig = inv(Omg)
    D   = np.random.multivariate_normal(np.zeros(num_var), Sig, num_smp)

    # partial correlation graph estimation
    problem  = csc_concord_fista(D, num_var=num_var, pMat=pMat,
                                 p_lambda=0.5, verbose=False, verbose_inn=True)
    Omg_est  = problem.solver_convset()

    # print('non-overlap nonzero entry count: ', np.count_nonzero(omega-invcov))




if __name__ == "__main__":

    test_synthetic()

