from cc_mrce import mrce
from cc_mrce import mrce_syn
from cscc_fista import cscc_fista

import numpy as np
import os, pickle, argparse, time
os.system("mode con cols=100")



def overall_objective(args, data, Omg_hat, B_hat):

    obj = - data.n * np.log(np.abs(Omg_hat.diagonal())).sum() \
          + (1/2) * np.trace( 
                        np.matmul(
                            np.matmul((data.Y - np.matmul(data.X, B_hat)).transpose(), 
                                      (data.Y - np.matmul(data.X, B_hat))), 
                            np.linalg.matrix_power(Omg_hat, 2))) \
          + (data.n * args.mrce_lambda / 2) * np.abs(B_hat).sum() \
          + (args.cscc_lambda / 2) * np.abs(Omg_hat).sum()
    return obj



class stat_syn(object):

    def __init__(self, n, Omg_hat=np.array([]), B_hat=np.array([]), 
                          X_ori=np.array([]),   Y_ori=np.array([])):

        self.n       = n
        self.Omg_hat = Omg_hat
        self.B_hat   = B_hat
        self.X_ori   = X_ori
        self.Y_ori   = Y_ori
        self.Y_hat   = np.array([])
        self.get_output()

    def get_output(self):

        self.Y_hat = np.matmul(self.X_ori, self.B_hat)

    def get_solution(self, B_hat):

        self.B_hat = B_hat

    def get_fpr(self, A_ori, A_hat):
        # compute False Positive Rate and True Positive Rate

        num_ori = np.count_nonzero(A_ori)
        num_hat = np.count_nonzero(A_hat) 

        num_TP = np.count_nonzero(np.logical_and.reduce([A_ori, A_hat]))
        num_FN = num_ori - num_TP
        num_FP = num_hat - num_TP
        num_TN = A_ori.size - num_ori

        FPR = num_FP / (num_FP + num_TN)
        TPR = num_TP / (num_TP + num_FN)
        return FPR, TPR

    def get_mse(self):
        mse = np.sum(
                    np.divide(
                        np.square(self.Y_ori-self.Y_hat).sum(axis=1), 
                        np.square(self.Y_ori).sum(axis=1)
                        )
                    ) / self.n
        return mse

    def get_mape(self):
        mape = np.sum(
                    np.divide(
                        np.abs(self.Y_ori-self.Y_hat).sum(axis=1), 
                        np.abs(self.Y_ori).sum(axis=1)
                        )
                    ) / self.n
        return mape

    def get_mpe(self):

        mpe = np.sum(
                    np.divide(
                        (self.Y_ori-self.Y_hat).sum(axis=1), 
                        self.Y_ori.sum(axis=1)
                        )
                    ) / self.n
        return mpe


def cscc_mrce(args):

    np.set_printoptions(formatter={'float': '{: 0.2e}'.format})
    
    if args.generate_synthetic:
        # generat a new synthetic dataset
        print("Generating synthetic dataset ...")
        data = mrce_syn(p = args.p, q = args.q, n = args.n, phi = args.phi, \
                err_type = args.err_type, rho = args.rho, pMat_noise = args.pMat_noise, \
                pct_nnz=args.pct_nnz, base_nnz=args.base_nnz, \
                success_prob_s1=args.success_prob_s1, success_prob_s2=args.success_prob_s2, \
                distr_type=args.distr_type, df=args.df)
        data.generate()
        with open(args.synthetic_dir, "wb") as pfile:
            pickle.dump(data, pfile) 
    else: 
        # use existing synthetic dataset
        print("Loading synthetic dataset ...")
        with open(args.synthetic_dir, 'rb') as pfile:
            data = pickle.load(pfile)

    if args.verbose:
        print("Groundtruth Omega:"); print(data.Omg)
        print('nonzero entry count: ', np.count_nonzero(data.Omg))
        print("Given pMat:"); print(data.pMat)
        print('nonzero entry count: ', np.count_nonzero(data.pMat))
        # input('... press any key to continue ...')

    # (estimate Omega)
    # partial correlation graph estimation
    if args.run_cscc:
        Omg_hat = np.zeros((data.q, data.q))
        record_label = os.path.splitext(os.path.basename(args.synthetic_dir))[0]
        problem = cscc_fista(D=data.Y-np.matmul(data.X, data.B), 
                    pMat=np.ones((data.q, data.q)), # pMat=data.pMat, 
                    num_var=data.q, 
                    step_type_out = args.cscc_step_type_out, 
                    const_ss_out = args.cscc_const_ss_out, c_outer=args.cscc_c_out,
                    p_gamma=args.cscc_gamma, p_lambda=args.cscc_lambda, p_tau=args.cscc_tau, 
                    MAX_ITR=args.cscc_max_itr, MAX_ITR_inn=args.cscc_max_itr_inn,
                    TOL=args.cscc_TOL, TOL_type=args.cscc_TOL_type,
                    TOL_inn=args.cscc_TOL_inn, c_inner=args.cscc_c_inn,
                    verbose=args.cscc_outer_verbose, 
                    verbose_inn=args.cscc_inner_verbose,
                    plot_in_loop=args.cscc_plot_in_loop,
                    no_constraints=args.no_constraints, inner_cvx_solver=args.inner_cvx_solver,
                    record_label=record_label, Omg_ori=data.Omg)
        Omg_hat, label = problem.solver_convset()
        if args.verbose:
            print("\n\n= = = Finished = = =\nGroundtruth Omega:"); print(data.Omg)
            print('nonzero entry count: ', np.count_nonzero(data.Omg))
            print("Inferred Omega:"); print(Omg_hat)
            print('nonzero entry count:', np.count_nonzero(Omg_hat))
    
    # if args.run_mrce:
        

    if args.cscc_pMat:
        pMat = data.pMat
    else:
        pMat = np.ones((data.q, data.q))

    if args.run_all:
        itr     = 0
        loop    = True
        B_hat   = np.zeros((data.p, data.q))
        Omg_hat = np.identity(data.q) # np.zeros((data.q, data.q))
        record_label = os.path.splitext(os.path.basename(args.synthetic_dir))[0]
        stat_obj = stat_syn(n=data.n, 
                            Omg_hat=Omg_hat, B_hat=B_hat, 
                            X_ori=data.X, Y_ori=data.Y)
        while loop:
            itr += 1
            # (estimate Omega)
            # partial correlation graph estimation
            t = time.time()
            problem = cscc_fista(D=data.Y-np.matmul(data.X, B_hat), 
                pMat=pMat, num_var=data.q, 
                step_type_out = args.cscc_step_type_out, 
                const_ss_out = args.cscc_const_ss_out, c_outer=args.cscc_c_out,
                p_gamma=args.cscc_gamma, p_lambda=args.cscc_lambda, p_tau=args.cscc_tau, 
                MAX_ITR=args.cscc_max_itr, 
                TOL=args.cscc_TOL, TOL_type=args.cscc_TOL_type,
                MAX_ITR_inn=args.cscc_max_itr_inn,
                TOL_inn=args.cscc_TOL_inn, c_inner=args.cscc_c_inn,
                verbose=args.cscc_outer_verbose, verbose_inn=args.cscc_inner_verbose,
                plot_in_loop=args.cscc_plot_in_loop,
                no_constraints=args.no_constraints, inner_cvx_solver=args.inner_cvx_solver,
                record=args.cscc_record, record_label=record_label, Omg_ori=data.Omg)
            Omg_hat, label = problem.solver_convset()
            # force threshold
            Omg_hat[np.abs(Omg_hat)<1e-3] = 0
            if args.verbose:
                print("= = = Omg-estimate itr-{0:d} finished in {1:.3f} s = = =".format(itr, time.time()-t))
                # print("Groundtruth Omega:"); print(data.Omg)
                print('Omg_ori nonzeros: ', np.count_nonzero(data.Omg))
                # print("Inferred Omega:"); print(Omg_hat)
                print('Omg_hat nonzeros: ', np.count_nonzero(Omg_hat))
                print('overall objective value:', overall_objective(args, data, Omg_hat, B_hat))

                # compute FPR
                FPR, TPR = stat_obj.get_fpr(data.Omg, Omg_hat)
                print('TPR: {0:.3f}, FPR: {1:.3f}'.format(TPR, FPR))

                # input('...press any key...')
                print("\n\n")
            
            # (estimate B) 
            # regression coefficient estimation
            t = time.time()
            problem = mrce(Omg=np.linalg.matrix_power(Omg_hat,2), 
                        lamb2=args.mrce_lambda, X=data.X, Y=data.Y,
                        step_type=args.mrce_step_type, const_ss=args.mrce_const_ss, 
                        c=0.5, p_tau=args.mrce_tau, alpha=1, 
                        TOL_ep=args.mrce_TOL, TOL_type=args.mrce_TOL_type,
                        max_itr=args.mrce_max_itr, 
                        verbose=args.mrce_verbose, verbose_plots=args.mrce_verbose_plots,
                        B_ori = data.B)
            B_hat = problem.fista_solver_B()
            stat_obj.get_solution(B_hat)
            stat_obj.get_output()
            if args.verbose:
                print("= = = B-estimate itr-{0:d} finished in {1:.3f} s = = =".format(itr, time.time()-t))
                # print("Groundtruth B:"); print(data.B)
                print('B_ori nonzeros: ', np.count_nonzero(data.B))
                # print("Inferred B:"); print(B_hat)
                print('B_hat nonzeros:', np.count_nonzero(B_hat))
                print('objective at ground-truth B: {:.3e}'.format(problem.likelihood_B(data.B)))
                print('overall objective value:', overall_objective(args, data, Omg_hat, B_hat))

                # compute FPR
                FPR, TPR = stat_obj.get_fpr(data.B, B_hat)
                print('TPR: {0:.3f}, FPR: {1:.3f}'.format(TPR, FPR))
                # compute MSE and ...
                print('MSE: {0:.3f}, MPE: {1:.3f}, MAPE: {2:.3f}'.format(
                            stat_obj.get_mse(), stat_obj.get_mpe(), stat_obj.get_mape()))

                # input('...press any key...')      
                print("\n\n")

            loop = itr <= args.max_itr
                

    return 



if __name__ == "__main__":

    """ 
    Example: generate a small synthetic dataset with
    Command: python cscc_merge.py --generate_synthetic --synthetic_dir 'data-utility/syn_pMat0.2.pkl' --p 7 --q 7 --n 50 --pMat_noise 0.2 

    Command: python cscc_merge.py --run_cscc --synthetic_dir 'data-utility/syn_pMat0.2.pkl'
    """

    parser = argparse.ArgumentParser(description='Arguments for CSCC-MRCE.')

    # Choose which algo to run
    parser.add_argument('--run_all', default=False, action='store_true',
                        help='Whether to run CSCC_MRCE to estimate Omg and B')
    parser.add_argument('--run_cscc', default=False, action='store_true',
                        help='Whether to run CSCC to estimate Omg')
    parser.add_argument('--run_mrce', default=False, action='store_true',
                        help='Whether to run MRCE to estimate B')
    parser.add_argument('--verbose', default=False, action='store_true',
                        help='Whether to display details of very-outer loop')
    parser.add_argument('--max_itr', type=int, default=20,
                        help='Maximum iteration of Omg-B loop')

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
    parser.add_argument('--distr_type', type=int, default=1, 
                        help='dataset generation: types of error distribution, 1 for multivariate normal distribution, 2 for multivariate t-distribution')
    parser.add_argument('--df', type=int, default=2, 
                        help='dataset generation: degrees of freedom for multivariate t-distribution')

    # Parameters of CSCC-Omg estimate
    parser.add_argument('--cscc_max_itr', type=int, default=20,
                        help='Maximum iteration of outer loop')
    parser.add_argument('--cscc_max_itr_inn', type=int, default=50,
                        help='Maximum iteration of inner loop')
    parser.add_argument('--cscc_step_type_out', type=int, default=3,
                        help='Type of step length setting')
    parser.add_argument('--cscc_const_ss_out', type=float, default=0.1,
                        help='Constant step length')
    parser.add_argument('--cscc_gamma', type=float, default=0.1,
                        help='gamma: penalty parameter in proximal operator')
    parser.add_argument('--cscc_lambda', type=float, default=0.2,
                        help='lambda: penalty parameter for l_1')
    parser.add_argument('--cscc_tau', type=float, default=0.2,
                        help='tau: step length')
    parser.add_argument('--cscc_TOL', type=float, default=1e-3,
                        help='Tolerance in outer loop')
    parser.add_argument('--cscc_TOL_type', type=int, default=0,
                        help='Convergence criterion type in outer loop')
    parser.add_argument('--cscc_TOL_inn', type=float, default=1e-2,
                        help='Tolerance in inner loop')
    parser.add_argument('--cscc_pMat', default=False, action='store_true', 
                        help='Whether to apply proximity mask')
    parser.add_argument('--cscc_c_out', type=float, default=0.5,
                        help='c: heuristic ratio in step length search in outer loop')
    parser.add_argument('--cscc_c_inn', type=float, default=0.5,
                        help='c: heuristic ratio in step length search in inner loop')


    # Debugging for CSCC-Omg estimate
    parser.add_argument('--cscc_inner_verbose', default=False, action='store_true',
                        help='Whether to display optimization updates of inner loop')
    parser.add_argument('--cscc_outer_verbose', default=False, action='store_true',
                        help='Whether to display optimization updates of outer loop')
    parser.add_argument('--inner_cvx_solver', default=False, action='store_true',
                        help='Use cvx solver in inner loop. Not recommended for high dimensional datasets such as HCP. Works ok on small synthetic datasets.')
    parser.add_argument('--no_constraints', default=False, action='store_true', 
                        help='Solve the problem with no constraints.')
    parser.add_argument('--cscc_record', default=False, action='store_true',
                        help='Whether to record meta-results in each iteration')
    parser.add_argument('--cscc_plot_in_loop', default=False, action='store_true',
                        help='Whether to plot meta-results of CSCC outer loop')
                        

    # Parameters of MRCE-B estimate
    parser.add_argument('--mrce_lambda', type=float, default=1,
                        help='lambda: penalty parameter for l_1')
    parser.add_argument('--mrce_tau', type=float, default=0.7,
                        help='tau: step length')
    parser.add_argument('--mrce_max_itr', type=int, default=50,
                        help='max_itr: set maximum iteration')
    parser.add_argument('--mrce_step_type', type=int, default=1,
                        help='Type of step length setting')
    parser.add_argument('--mrce_const_ss', type=float, default=0.1,
                        help='Constant step length')
    parser.add_argument('--mrce_TOL', type=float, default=1e-3,
                        help='Tolerance in B-estimate')
    parser.add_argument('--mrce_TOL_type', type=int, default=0,
                        help='Convergence criterion type in B-estimate')

    # Debugging for MRCE-B estimate
    parser.add_argument('--gt_init', default=False, action='store_true',
                        help='Use groundtruth solution as the initial position')
    parser.add_argument('--mrce_verbose', default=False, action='store_true',
                        help='Whether to display details of optimization updates')
    parser.add_argument('--mrce_verbose_plots', default=False, action='store_true',
                        help='Whether to display plots')

    args = parser.parse_args()
    cscc_mrce(args)