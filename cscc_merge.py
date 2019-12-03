from cc_mrce import mrce_syn
from cc_mrce import mrce
from cscc_fista import cscc_fista

import numpy as np
import os, pickle, argparse


def cscc_mrce(args):

    print("Loading synthetic dataset ...")

    if args.generate_synthetic:
        # generat a new synthetic dataset
        data = mrce_syn(p = args.p, q = args.q, n = args.n, phi = args.phi, \
            err_type = args.err_type, rho = args.rho, pMat_noise = args.pMat_noise)
        data.generate()
        with open(args.synthetic_dir, "wb") as pfile:
            pickle.dump(data, pfile) 
    else: 
        # use existing synthetic dataset
        with open(args.synthetic_dir, 'rb') as pfile:
            data = pickle.load(pfile)

    if args.mrce_verbose:
        print("Groundtruth Omega:"); print(data.Omg)
        print('nonzero entry count: ', np.count_nonzero(data.Omg))
        print("Given pMat:"); print(data.pMat)
        print('nonzero entry count: ', np.count_nonzero(data.pMat))

    if args.run_algo:
        loop    = True
        B_hat   = np.zeros((data.p, data.q))
        Omg_hat = np.zeros((data.q, data.q))
        record_label = os.path.splitext(os.path.basename(args.synthetic_dir))[0]
        while loop:
            # (estimate Omega)
            # partial correlation graph estimation
            problem = cscc_fista(D=data.Y-np.matmul(data.X, B_hat), 
                        pMat=data.pMat, num_var=data.q, 
                        step_type_out = args.cscc_step_type_out, const_ss_out = args.cscc_const_ss_out, 
                        p_gamma=args.cscc_gamma, p_lambda=args.cscc_lambda, p_tau=args.cscc_tau, 
                        MAX_ITR=args.cscc_max_itr, 
                        TOL=args.cscc_TOL, TOL_inn=args.cscc_TOL_inn,
                        verbose=args.cscc_outer_verbose, verbose_inn=args.cscc_inner_verbose,
                        no_constraints=args.no_constraints, inner_cvx_solver=args.inner_cvx_solver,
                        record_label=record_label)
            Omg_hat, label = problem.solver_convset()
            if args.mrce_verbose:
                print("\n\n= = = Finished = = =\nGroundtruth Omega:"); print(data.Omg)
                print('nonzero entry count: ', np.count_nonzero(data.Omg))
                print("Inferred Omega:"); print(Omg_hat)
                print('nonzero entry count:', np.count_nonzero(Omg_hat))
            
            # (estimate B) 
            # regression coefficient estimation
            problem = mrce(Omg=np.linalg.matrix_power(Omg_hat,2),
                        lamb2=args.mrce_lambda, X=data.X, Y=data.Y,
                        step_type=args.mrce_step_type, const_ss=args.mrce_const_ss, 
                        c=0.5, p_tau=args.mrce_tau, alpha=1, 
                        TOL_ep=0.05, max_itr=args.mrce_max_itr, 
                        verbose=args.mrce_verbose, verbose_plots=args.mrce_verbose_plots)
            if args.mrce_verbose:
                print('objective at ground-truth B: {:.3e}'.format(problem.likelihood_B(data.B)))
                input('...press any key...')
            B_hat   = problem.fista_solver_B()

    return 



if __name__ == "__main__":

    """ Example: generate a small synthetic dataset with
    python cscc_merge.py --generate_synthetic --synthetic_dir 'data-utility/syn_pMat0.2.pkl' --p 7 --q 7 --n 50 --pMat_noise 0.2 """

    parser = argparse.ArgumentParser(description='Arguments for CSCC-MRCE.')

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
    parser.add_argument('--cscc_TOL_inn', type=float, default=1e-2,
                        help='Tolerance in inner loop')

    # Debugging for CSCC-Omg estimate
    parser.add_argument('--cscc_inner_verbose', default=False, action='store_true',
                        help='Whether to display optimization updates of inner loop')
    parser.add_argument('--cscc_outer_verbose', default=True, action='store_true',
                        help='Whether to display optimization updates of outer loop')
    parser.add_argument('--inner_cvx_solver', default=False, action='store_true',
                        help='Use cvx solver in inner loop.')
    parser.add_argument('--no_constraints', default=False, action='store_true', 
                        help='Solve the problem with no constraints.')
    parser.add_argument('--run_algo', default=False, action='store_true',
                        help='Whether to run the algorithm')

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

    # Debugging for MRCE-B estimate
    parser.add_argument('--gt_init', default=False, action='store_true',
                        help='Use groundtruth solution as the initial position')
    parser.add_argument('--mrce_verbose', default=False, action='store_true',
                        help='Whether to display details of optimization updates')
    parser.add_argument('--mrce_verbose_plots', default=False, action='store_true',
                        help='Whether to display plots')

    args = parser.parse_args()
    cscc_mrce(args)