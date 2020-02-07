import numpy as np
import cvxpy as cvx

from scipy import sparse
from scipy.stats import ortho_group

import seaborn as sns
import matplotlib.pyplot as plt
import sys, os, pickle, argparse, math, time, pickle

from matplotlib.pyplot import cm


def plot_grad_hist(G, G_wnorm, prefix='_0'):

    plt.figure()
    sns.distplot(np.array(G.T)[0], color="skyblue", label="nabla_h")
    sns.distplot(np.array(G_wnorm.T)[0], color="red", label="nabla_h+subgrad(l_1)")
    plt.title('histogram of subgradient, entrywise')
    plt.legend()
    plt.savefig("record/img/temp_grad_hist"+prefix+".png", dpi=200)
    plt.close()
    return 


class mrce(object):
    """ MRCE: multivariate regression with covariance estimation """

    def __init__(self, X, Y, Omg, lamb2, TOL_ep, \
        max_itr=100, B_init=np.array([]), B_ori=np.array([]), \
        verbose=True, verbose_plots=False, TOL_type=1, \
        u_cache=False, matrix_form=True, stochastic=False, \
        step_type=3, c=0.9, alpha=1, p_tau=0.7, const_ss=0.2):

        self.X = X
        self.Y = Y
        # self.Omg = Omg

        self.lamb2 = lamb2
        self.p = self.X.shape[1]
        self.q = self.Y.shape[1]
        self.n = self.X.shape[0]
        self.TOL_ep = TOL_ep
        self.max_itr = max_itr
        
        self.matrix_form = matrix_form
        self.u_cache = u_cache
        self.verbose = verbose
        self.verbose_plots = verbose_plots
        self.stochastic = stochastic

        # set up parameters for FISTA
        self.c         = c
        self.alpha     = alpha
        self.p_tau     = p_tau
        self.step_type = step_type
        self.const_ss  = const_ss
        self.TOL_type  = TOL_type

        # initialization
        if self.verbose: 
            print("\n\n\n = = = MRCE B-step Initialization = = = ")
        # compute S = X'X
        t = time.time()
        self.S = np.matmul(self.X.transpose(), self.X)
        if self.verbose: 
            print("MRCE: S computed in {:.2e} s".format(time.time()-t))
        # compute H = X'Y*Omega
        t = time.time()
        temp = np.matmul(self.X.transpose(), self.Y)
        if self.verbose: 
            print("MRCE: H_0 computed in {:.2e} s".format(time.time()-t))

        # compute ridge estimate of B
        self.B_ridge = self.ridge_estimate()
        self.B_ori   = B_ori

        # set initial value of B
        if B_init.size == 0:
            # self.B_init = np.random.normal(0,1, (self.p, self.q))
            self.B_init = self.B_ridge
        else:
            self.B_init = B_init
        
        if Omg.size == 0:
            # init_E = (self.Y - np.matmul(self.X, self.B_init))
            # E_S = init_E - np.tile(init_E.mean(axis=0),(init_E.shape[0],1))
            # self.Omg = np.linalg.inv((E_S.transpose() @ E_S) / (E_S.shape[0] - 1))
            self.Omg = np.identity(self.q)
        else:
            self.Omg = Omg

        t = time.time()
        self.H = np.matmul(temp, self.Omg)
        if self.verbose: 
            print("MRCE: H computed in {:.2e} s".format(time.time()-t))
        
        if u_cache:
            self.SOmg = np.empty([self.p, self.q])

    def ridge_estimate(self):
        """ ridge estimate of B that is usedfor convergence test: 
            B = (X'*X+lamb2*I)^(-1)*X'*Y """

        try:
           temp = np.linalg.inv(self.S + self.lamb2 * np.eye(self.p))
        except np.linalg.LinAlgError as err:
            if 'Singular matrix' in str(err):
                print('Rdige estimate: singular matrix causes inverse failure. ')
            else:
                raise
        B_ridge = np.matmul(temp, np.matmul(self.X.transpose(), self.Y)) 
        return B_ridge


    def likelihood_B(self, B):
        """ compute the objective value of B-step in MRCE """

        return np.trace( (1/self.n) * 
                    np.matmul(
                        np.matmul((self.Y-np.matmul(self.X,B)).transpose(), 
                                  (self.Y-np.matmul(self.X,B))), 
                        self.Omg)
                ) + self.lamb2 * np.abs(B).sum()

    def likelihood_B_wonorm(self, B):
        """ compute the objective value of B-step in MRCE """

        return np.trace( (1/self.n) * 
                    np.matmul(
                        np.matmul((self.Y-np.matmul(self.X,B)).transpose(), 
                                  (self.Y-np.matmul(self.X,B))), 
                        self.Omg)
                )


    def cvx_solver_B(self):
        """ cvx solver for B-update """

        print(self.p)
        print(self.q)
        B    = cvx.Variable((self.p,self.q))
        loss = cvx.trace( (1/self.n) * 
                    cvx.matmul(
                        self.Omg,
                        cvx.matmul( (self.Y-cvx.matmul(self.X,B)).T,
                                    (self.Y-cvx.matmul(self.X,B)))   
                    )
                ) + self.lamb2 * cvx.norm(B, 1)
        obj = cvx.Minimize(loss)
        prob = cvx.Problem(obj)
        prob.solve(verbose=True)

        if self.verbose:
            print("obj_B: {:.3e}".format(self.likelihood_B(B.value)))
            print('nonzero entry count: ', np.count_nonzero(B.value))
        return


    def fista_solver_B(self):
        """ FISTA for B update """

        # print('\n\n\nobjective at B_ridge: {:.3e}'.format(self.likelihood_B(self.B_ridge)))
        # print('nonzero B_ridge-entry count: ', np.count_nonzero(self.B_ridge))

        tau_n  = self.p_tau
        alpha  = self.alpha

        Lambda = self.lamb2 * np.ones(self.B_init.shape)
        if self.step_type == 3:
            tau_n = self.const_ss

        # Omega initial likelihood
        B  = self.B_init.copy()
        h  = self.likelihood_B_wonorm(B)
        f  = h + self.lamb2 * np.abs(B).sum()

        # Theta & initial gradient
        Th    = self.B_init.copy()
        XYOmg = np.matmul(np.matmul(self.X.transpose(), self.Y), self.Omg)
        G     = (2/self.n) * (np.matmul(self.S, np.matmul(Th, self.Omg)) - XYOmg)

        # plotting data
        if self.verbose_plots:
            plot_data = {}
            plot_data['x']     = list()
            plot_data['y(B)']  = list(); plot_data['y(Th)'] = list();   plot_data['y(A)'] = list()
            plot_data['L1(B)'] = list(); plot_data['L1(Th)'] = list();  plot_data['L1(A)'] = list()
            plot_data['h(B)']  = list(); plot_data['h(Th)'] = list();   plot_data['h(A)'] = list()
            plot_data['tau']   = list(); 
            plot_data['subg']  = list();   
            plot_data['subg_diff'] = list()

        # looping for optimization steps
        loop = True
        itr  = 0
        h_n  = 0.0
        Q_n  = 0.0
        f_n  = f
        while loop:
            itr_back = 0
            tau      = tau_n
            if self.verbose: 
                print("\n = = = MRCE: B iteration "+str(itr)+" = = = ")
 
            # constant step length
            if self.step_type == 3:
                G_n = (2/self.n) * (np.matmul(self.S, np.matmul(Th, self.Omg)) - XYOmg)
                A_n = Th - tau * G_n
                B_n = np.sign(A_n) * np.maximum(abs(A_n)-tau*Lambda, 0.0)
                h_n = self.likelihood_B_wonorm(B_n)

            # looping for adaptive step length as backtacking line search
            else:
                while True:
                    if itr_back != 0:
                        tau = tau * self.c
                    if self.verbose: 
                        print("- - - line-search iteration "+str(itr_back)+" - - - ")
                    G_n = (2/self.n) * (np.matmul(self.S, np.matmul(Th, self.Omg)) - XYOmg)
                    A_n = Th - tau * G_n
                    B_n = np.sign(A_n) * np.maximum(abs(A_n)-tau*Lambda, 0.0)

                    # check backtracking condition
                    Step = B_n - Th
                    Q_n  = self.likelihood_B_wonorm(Th) \
                        + (Step*G).sum() + (1/(2*tau))*(np.linalg.norm(Step)**2) \
                        + np.abs(B_n).sum()
                    h_n  = self.likelihood_B_wonorm(B_n)
                    if h_n > Q_n: # sufficient descent condition
                        itr_back += 1
                    else:
                        break
                # end of (backtrack line-search)
                if self.verbose:
                    print("tau: {:.3e}".format(tau_n))
            
            if self.verbose_plots:
                plot_data['tau'].append(tau)

            
            # check gradient
            subg_diff = self.likelihood_B_wonorm(Th) - self.likelihood_B_wonorm(A_n) \
                        + ((A_n-Th)*G_n).sum()
            if self.verbose_plots:
                plot_data['subg_diff'].append(subg_diff)

            # check function value
            f_n = h_n + self.lamb2 * np.abs(B_n).sum()
            if self.verbose:
                print("MRCE: check gradient correctness {:.3e}".format(subg_diff))
                print("- - - problem solution UPDATED - - -\n" + \
                    " 1st term (loss): "+"{:.3e}".format(h_n) + \
                    "\n 2nd term (regularizer): "+"{:.3e}".format(f_n-h_n) + \
                    "\n objective: "+"{:.3e}".format(f_n))

            if self.verbose_plots:
                plot_data['x'].append(itr) 
                plot_data['y(A)'].append(self.likelihood_B(A_n))
                plot_data['y(B)'].append(self.likelihood_B(B_n))
                plot_data['y(Th)'].append(self.likelihood_B(Th))
                plot_data['h(A)'].append(self.likelihood_B_wonorm(A_n))
                plot_data['h(B)'].append(self.likelihood_B_wonorm(B_n))
                plot_data['h(Th)'].append(self.likelihood_B_wonorm(Th))
                plot_data['L1(A)'].append(self.lamb2 * np.abs(A_n).sum())  
                plot_data['L1(B)'].append(self.lamb2 * np.abs(B_n).sum())  
                plot_data['L1(Th)'].append(self.lamb2 * np.abs(Th).sum())

            # - - - - FISTA - - - - 
            # momentum update step
            alpha_n = (1 + math.sqrt(1 + 4*(alpha**2)))/2
            Th  = B_n + ((alpha-1)/alpha_n) * (B_n - B)
            # update gradient
            G_n = (2/self.n) * (np.matmul(self.S, np.matmul(Th, self.Omg)) - XYOmg)
            # update tau for next opt iteration
            if self.step_type == 0:
                tau_n = 1
            elif self.step_type == 1:
                tau_n = tau
            elif self.step_type == 2:
                tau_n = (Step * Step).sum() / (Step * (G_n - G)).sum()
                tau_n = tau if tau_n < 0.0 else tau_n

            # compute stopping criterion
            if self.TOL_type == 0:
                tmp       = G_n + np.sign(B_n) * Lambda # sign(Omg_n) or sign(Th)
                subg      = np.sign(G_n) * np.maximum(abs(G_n) - Lambda, 0.0)
                subg[B_n != 0] = tmp[B_n != 0]
                # cur_err   = np.linalg.norm(G_n)
                cur_err   = np.linalg.norm(subg) / np.linalg.norm(B_n)
                # cur_err   = np.linalg.norm(B_n + np.sign(B_n) * Lambda) / np.linalg.norm(B_n)
                if self.verbose_plots:
                    plot_data['subg'].append(cur_err)
                if self.verbose:
                    print("subg norm:{:.3e}".format(np.linalg.norm(subg)))
            elif self.TOL_type == 1:
                # print("f_n:{0:.3e}, f:{1:.3e}".format(f_n,f))
                cur_err   = np.abs((f_n - f) / f)

            if self.verbose: 
                print("error: "+"{:.3e}".format(cur_err)+\
                    "\nh function value:"+"{:.3e}".format(h_n))
                print('nonzero entry count of B: ', np.count_nonzero(B_n))
                if np.isnan(h_n): sys.exit()
            
            # update for next iteration
            alpha = alpha_n
            B     = B_n
            h     = h_n
            f     = f_n
            G     = G_n
            itr  += 1

            if self.verbose_plots:
                plt.figure(1,figsize=(12, 7))
                plt.subplot(231); plt.title('overall objective')
                plt.plot(plot_data['x'], plot_data['y(Th)'], 'c.-', label='y(Th)')
                # plt.plot(plot_data['x'], plot_data['y(A)'], 'r.-', label='y(A)')
                plt.plot(plot_data['x'], plot_data['y(B)'], 'b.-', label='y(B)')
                plt.yscale('log'); plt.show(block=False); plt.pause(0.01)
                if itr == 1: plt.legend(); 

                plt.subplot(232); plt.title('data fidelity term')
                plt.plot(plot_data['x'], plot_data['h(Th)'], 'c.--', label='h(Th)')
                # plt.plot(plot_data['x'], plot_data['h(A)'], 'r.--', label='h(A)')
                plt.plot(plot_data['x'], plot_data['h(B)'], 'b.--', label='h(B)')
                plt.yscale('log'); plt.show(block=False); plt.pause(0.01)
                if itr == 1: plt.legend(); 

                plt.subplot(233); plt.title('penalty term')
                plt.plot(plot_data['x'], plot_data['L1(Th)'], 'c.--', label='L1(Th)')
                # plt.plot(plot_data['x'], plot_data['L1(A)'], 'r.--', label='L1(A)')
                plt.plot(plot_data['x'], plot_data['L1(B)'], 'b.--', label='L1(B)')
                plt.yscale('log'); plt.show(block=False); plt.pause(0.01)
                if itr == 1: plt.legend(); 

                plt.subplot(234); plt.title('step size')
                plt.plot(plot_data['x'], plot_data['tau'], 'r.--')
                plt.show(block=False); plt.pause(0.01)

                if self.TOL_type == 0:
                    plt.subplot(235); plt.title('norm of subgradient')
                    plt.plot(plot_data['x'], plot_data['subg'], 'k.--')
                    plt.yscale('log'); plt.show(block=False); plt.pause(0.01)

                plt.subplot(236); plt.title('check correctness of gradient')
                plt.plot(plot_data['x'], plot_data['subg_diff'], 'b.--')
                plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
                plt.show(block=False); plt.pause(0.01)
            
                if np.mod(itr, 10) == 0:
                    G = (2/self.n) * (np.matmul(self.S, np.matmul(B, self.Omg)) - XYOmg)
                    G_wnorm = G + self.lamb2 * np.sign(B)
                    plot_grad_hist(G, G_wnorm, '_'+str(itr))
                
            # check termination condition:
            loop = itr < self.max_itr and cur_err > self.TOL_ep
         # end of (while loop)
        
        print("MRCE: B estimate ends at iteration {0:d}".format(itr-1))

        if self.verbose_plots:
            plt.show(block=True)
        # if self.verbose:
        #     print("Inferred B:")
        #     print(B_n)
        #     print("Groundtruth B:")
        #     print(self.B_ori)
        return B_n



    def coordinate_solver_B(self):
        """ cyclical-coordinate descent algorithm """

        B = self.B_init.copy()
        B_n = self.B_init.copy()

        if self.verbose:
            plt.ion() ## Note this correction
            plt.figure()

        plot_data = {}
        plot_data['x'] = list()
        plot_data['y'] = list()

        loop = True
        itr = 0
        while loop:

            if self.verbose: 
                print("\n\n\n\n = = = B iteration "+str(itr)+" = = = ")
            
            if not self.u_cache:
                t = time.time()
                U = np.matmul(np.matmul(self.S, B), self.Omg)
                print("MRCE: U computed in {:.2e} s".format(time.time()-t))
            
            if not self.matrix_form:
                if not self.stochastic:
                    print("cyclically updating B_n[r,c] ...")
                    for r in range(self.p):
                        for c in range(self.q):
                            d = self.S[r,r] * self.Omg[c,c]
                            U_n = np.matmul(np.matmul(self.S, B_n), self.Omg)
                            temp = B_n[r,c] + (self.H[r,c] - U_n[r,c]) / d
                            sthresh = abs(temp) - (self.n*self.lamb2)/d
                            sthresh = 0 if sthresh < 0 else sthresh
                            B_n[r,c] = np.sign(temp) * sthresh
                else:
                    print("randomly updating B_n[r,c] ...")
                    for i in range(1000):
                        r = np.random.randint(self.p)
                        c = np.random.randint(self.q)
                        d = self.S[r,r] * self.Omg[c,c]
                        U_n = np.matmul(np.matmul(self.S, B_n), self.Omg)
                        temp = B_n[r,c] + (self.H[r,c] - U_n[r,c]) / d
                        sthresh = abs(temp) - (self.n*self.lamb2)/d
                        sthresh = 0 if sthresh < 0 else sthresh
                        B_n[r,c] = np.sign(temp) * sthresh
            else:
                D = np.matmul(np.diag(self.S)[:, np.newaxis], \
                              np.diag(self.Omg)[:, np.newaxis].transpose())
                temp = B + np.divide(self.H - U, D)
                sthresh = np.abs(temp) - (self.n * self.lamb2) / D
                sthresh[sthresh < 0] = 0
                B_n = np.sign(temp) * sthresh

            f_n = self.likelihood_B(B_n)
            plot_data['x'].append(itr) 
            plot_data['y'].append(f_n) 
            if self.verbose:
                plt.plot(plot_data['x'], plot_data['y'], 'bo-')
                plt.show(block=False)
                plt.pause(0.0001)

            print("computing stopping criterion ...")
            err_B = np.abs(B - B_n).sum()
            norm_B = np.abs(self.B_ridge).sum()
            stop =  err_B < self.TOL_ep *  norm_B
            itr  = itr + 1
            B    = B_n

            if self.verbose:
                print("obj_B: {:.3e}".format(f_n))
                print("err_B: {0:.3e}, norm_B: {1:.3e}".format(err_B, norm_B))
                print('nonzero entry count: ', np.count_nonzero(B))

            # if stop or itr > self.max_itr:
            #     break
            if itr > self.max_itr:
                break

        plt.show(block=True)
        return B_n 



class mrce_syn(object):
    """ Generate synthetic dataset according to MRCE paper """

    def __init__(self, p = 100, q = 100, n = 50, phi = 0.7, \
        err_type = 0, rho = 0.5, Hurst = 0.9, success_prob_s1 = 0.2, success_prob_s2 = 0.2, \
        pct_nnz = 0.2, base_nnz = 0.7, pMat_noise = 0):

        self.p = p
        self.n = n
        self.q = q 

        # - - - baseline covariance value of X dimensions
        self.phi = phi
    
        # - - - Types of error covariance
        # 0 -> AR(1)
        # 1 -> Fractional Gaussian Noise (FGN) 
        self.err_type = err_type

        # - - - baseline covariance value in AR(1)
        # ranging from 0 to 0.9
        self.rho = rho

        # - - - Hurst parameter in FGN
        # cov and invcov are both dense
        # ranging from 0.5 to 1
        # 0.5 -> i.i.d sequence
        # 1.0 -> perfectly correlated 
        self.Hurst = Hurst
        self.success_prob_s1 = success_prob_s1
        self.success_prob_s2 = success_prob_s2

        self.X = np.empty([self.n, self.p])
        self.Y = np.empty([self.n, self.q])
        self.E = np.empty([self.n, self.q])
        self.B = np.empty([self.p, self.q])
        self.Sigma_X = np.empty([self.p, self.p])
        self.Sigma_E = np.empty([self.q, self.q])

        # - - - Parameters for err_type = 2
        self.pct_nnz  = pct_nnz   # percentage of non-zero entries in L matrix
        self.base_nnz = base_nnz  # base value of non-zero entries in L matrix

        # - - - noise ratio of noise in pMat
        self.pMat_noise = pMat_noise
        
        return

    def generate(self):

        # generate Sigma_X (p x p)
        for i in range(self.p):
            for j in range(self.p):
                self.Sigma_X[i,j] = np.power(self.phi, np.abs(i-j)) 
        # generate X (n x p)
        self.X = np.random.multivariate_normal(np.zeros(self.p), self.Sigma_X, self.n)

        # generate Sigma_E (q x q) and Omg (q x q)
        if self.err_type == 0: 
            # AR(1) error covariance
            for i in range(self.q):
                for j in range(self.q):
                    self.Sigma_E[i,j] =  np.power(self.rho, np.abs(i-j))
            # generate Omg from Sigma_E
            self.Omg  = np.linalg.inv(self.Sigma_E)
        elif self.err_type == 1:
            # Fractional Gaussian Noise (FGN)
            for i in range(self.q):
                for j in range(self.q):
                    self.Sigma_E[i,j] = 0.5 * (np.power(np.abs(i-j)+1, 2*self.Hurst) 
                                            -2*np.power(np.power(i-j), 2*self.Hurst)
                                            +  np.power(np.abs(i-j)-1, 2*self.Hurst) )
            self.Omg  = np.linalg.inv(self.Sigma_E)
        elif self.err_type == 2:
            # Randomly select a certain number of edges
            #       as non-zeros in partial correlation graph.
            # Create sparse symmetric positive definite matrix:
            #       Every positive-definite matrix has a Cholesky decomposition 
            #       that takes the form LL' where L is lower triangular, 
            #       so sample L and compute a positive-definite matrix from it. 
            #       If L is sparse then LL' is also sparse. 
            #       Make sure L is less sparse then what you want your final matrix to be.
            Spr = sparse.random(self.q, self.q, density=self.pct_nnz).A
            Spr[Spr != 0] = Spr[Spr != 0] * (1 - self.base_nnz) + self.base_nnz
            print("... nonzeros of triu-Spr: " + str(np.count_nonzero(Spr)))
            Chol = np.tril(Spr)
            np.fill_diagonal(Chol, 1)
            self.Omg  = Chol @ Chol.transpose()
            print("... nonzeros of triu-Omg: " + str((np.count_nonzero(self.Omg)-self.q)/2.0))
            self.Omg = self.Omg / np.max(self.Omg)
            np.fill_diagonal(self.Omg, 1)
            print("... positive definiteness check: " + str(np.all(np.linalg.eigvals(self.Omg) > 0)))
            self.Sigma_E = np.linalg.inv(self.Omg)
        
        # generate error matrix E (n x q)
        # option 1: generate multivariate gaussian noise
        self.E = np.random.multivariate_normal(np.zeros(self.q), self.Sigma_E, self.n)
        # option 2: generate t-distributed noise


        # generate B
        W = np.random.normal(0, 1, (self.p, self.q))
        K = np.random.binomial(n=1, p=self.success_prob_s1, size=(self.p, self.q))
        Q_idx = np.random.binomial(n=1, p=self.success_prob_s2, size=self.p)
        Q = np.zeros((self.p, self.q))
        Q[np.nonzero(Q_idx)[0],:] = 1
        self.B = W * K * Q  # element-wise product
        print("... ground-truth B contains {0:d} non-zeros ...".format(np.count_nonzero(self.B)))

        # generate Y
        self.Y = np.matmul(self.X, self.B) + self.E

        # create non-zero mask
        self.pMat = self.Omg.copy()
        self.pMat[self.pMat != 0] = 1
        num_nz = np.count_nonzero(self.pMat)
        print("... ground-truth Omg contains {0:d} non-zero entries, {1:d} off-diag ...".format(num_nz, num_nz-self.q))
        
        # add more feasible entries to pMat mask
        if self.pMat_noise != 0:
            num_noise = int(np.floor(num_nz * self.pMat_noise/2))
            print("... sampling {0:d} entries in pMat ...".format(2 * num_noise))
            idx_r = np.random.randint(self.q, size=num_noise)
            idx_c = np.random.randint(self.q, size=num_noise)
            self.pMat[idx_r, idx_c] = 1
            self.pMat = self.pMat + self.pMat.transpose()
            self.pMat[self.pMat != 0] = 1     
            num_nz2 = np.count_nonzero(self.pMat)
            print("... adding {0:d} feasible entries in pMat ...".format(num_nz2-num_nz))
            print("... symmetric check: {0} ...".format(np.allclose(self.pMat, self.pMat.T, rtol=1e-5, atol=1e-8)))

        return



def test(args):


    if args.generate_synthetic:
        # generat a new synthetic dataset
        data = mrce_syn(p = 500, q = 1000, n = 200, 
                        err_type = 0, rho = 0.5, phi = 0.7, 
                        pct_nnz=args.pct_nnz, base_nnz=args.base_nnz)
        data.generate()
        with open(args.synthetic_dir, "wb") as pfile:
            pickle.dump(data, pfile) 
    else: 
        # use existing synthetic dataset
        with open(args.synthetic_dir, 'rb') as pfile:
            data = pickle.load(pfile)

    if 'data' in locals():
        print('= = = dataset loaded = = =')
        print('sample size: {0:d}'.format(data.n))
        print('input-output dimensions: {0:d}-{1:d}'.format(data.X.shape[1], data.Y.shape[1]))
        print('nonzero B-entry count: ', np.count_nonzero(data.B))

        Omg_ori = np.linalg.inv(data.Sigma_E)
        print('\nnonzero Omg-entry count: ', np.count_nonzero(Omg_ori))
        print('check non-negative definiteness: '+str(np.all(np.linalg.eigvals(Omg_ori) >= 0))+'\n')

        XYOmg = np.matmul(np.matmul(data.X.transpose(), data.Y), Omg_ori)
        G = (2/data.n) * (np.matmul(np.matmul(data.X.transpose(), data.X), np.matmul(data.B, Omg_ori)) - XYOmg)
        G_wnorm = G + args.p_lambda * np.sign(data.B)
        # plot_grad_hist(G, G_wnorm)

    B0 = data.B if args.gt_init else np.array([])
        
    if args.CD:
        problem = mrce(X=data.X, Y=data.Y, Omg=Omg_ori, lamb2=args.p_lambda, TOL_ep=0.05, matrix_form=args.max_itr, stochastic=False, max_itr=args.max_itr, B_init=B0,verbose=args.verbose, verbose_plots=args.verbose_plots)
        print('objective at ground-truth B: {:.3e}'.format(problem.likelihood_B(data.B)))
        input('...press any key...')
        B = problem.coordinate_solver_B()
    elif args.FISTA:
        problem = mrce(X=data.X, Y=data.Y, Omg=Omg_ori, lamb2=args.p_lambda, TOL_ep=0.05, max_itr=args.max_itr, step_type=args.step_type, c=0.8, p_tau=args.p_tau, alpha=1, const_ss=args.const_ss, B_init=B0, verbose=args.verbose, verbose_plots=args.verbose_plots)
        print('objective at ground-truth B: {:.3e}'.format(problem.likelihood_B(data.B)))
        input('...press any key...')
        B = problem.fista_solver_B()
        with open('record/solution_B.pkl', 'wb') as pfile:
            pickle.dump(B, pfile)

    if 'problem' in locals():
        print('\n\n\nobjective at ground-truth B: {:.3e}'.format(problem.likelihood_B(data.B)))
        print('nonzero B-entry count: ', np.count_nonzero(data.B))

    return


if __name__ == "__main__":

    # example: python cc_mrce.py --synthetic_dir 'data-utility/synB-500.pkl' --FISTA --verbose --verbose_plots
    
    parser = argparse.ArgumentParser(description='Arguments for MRCE B-update.')

    # Pick or generate a dataset
    parser.add_argument('--generate_synthetic', default=False, action='store_true',
                        help='Whether to generate a new synthetic dataset')
    parser.add_argument('--synthetic_dir', type=str, default='data-utility/synB.pkl',
                        help='File path to the new synthetic dataset')
    parser.add_argument('--p', type=int, default=500, 
                        help='dataset generation: input dimension')
    parser.add_argument('--q', type=int, default=1000, 
                        help='dataset generation: output dimension')
    parser.add_argument('--n', type=int, default=200, 
                        help='dataset generation: sample size')
    parser.add_argument('--phi', type=float, default=0.7, 
                        help='dataset generation: baseline covariance value of multivariate inputs')
    parser.add_argument('--err_type', type=int, default=0, 
                        help='dataset generation: types of error covariance, 0 for AR(1), 1 for Fractional Gaussian Noise (FGN)')
    parser.add_argument('--rho', type=float, default=0.5, 
                        help='dataset generation AR(1): baseline covariance value, from 0 to 0.9')
    parser.add_argument('--pMat_noise', type=float, default=0, 
                        help='dataset generation: additional noise ratio in pMat, from 0 to 1')
    parser.add_argument('--Hurst', type=float, default=0.9,
                        help='dataset generation FGN: Hurst parameter')
    parser.add_argument('--success_prob_s1', type=float, default=0.2,
                        help='dataset generation FGN:  Bernoulli draws for K entries')
    parser.add_argument('--success_prob_s2', type=float, default=0.2,
                        help='dataset generation FGN:  Bernoulli draws for Q rows')
    parser.add_argument('--pct_nnz', type=float, default=0.2, 
                        help='dataset generation Chol: percentage of non-zero entries in L matrix')
    parser.add_argument('--base_nnz', type=float, default=0.7, 
                        help='dataset generation Chol: base value of non-zero entries in L matrix')

    # Pick a solver
    parser.add_argument('--CD', default=False, action='store_true',
                        help='Apply coordinate descent algorithm')
    parser.add_argument('--FISTA', default=False, action='store_true',
                        help='Apply FISTA algorithm')

    # Parameters of algorithm
    parser.add_argument('--p_lambda', type=float, default=1,
                        help='lambda: penalty parameter for l_1')
    parser.add_argument('--p_tau', type=float, default=0.7,
                        help='tau: step length')
    parser.add_argument('--max_itr', type=int, default=50,
                        help='max_itr: set maximum iteration')
    parser.add_argument('--step_type', type=int, default=1,
                        help='Type of step length setting')
    parser.add_argument('--const_ss', type=float, default=0.1,
                        help='Constant step length')

    # For debugging
    parser.add_argument('--gt_init', default=False, action='store_true',
                        help='Use groundtruth solution as the initial position')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Whether to display details of optimization updates')
    parser.add_argument('--verbose_plots', default=False, action='store_true',
                        help='Whether to display plots')
    
    args = parser.parse_args()

    test(args)