python cscc_merge.py --run_all --verbose \
--cscc_lambda 0.5 --mrce_lambda 0.3 --max_itr 50 \
--synthetic_dir 'data-utility/syndata_n50p20q20.pkl' \
--cscc_TOL 1e-3 --cscc_TOL_type 1 \
--cscc_max_itr 100 \
--cscc_step_type_out 1 --cscc_const_ss_out 0.02 \
--cscc_tau 1 --cscc_c_out 0.5 --cscc_gamma 1 \
--cscc_TOL_inn 1e-4 --cscc_max_itr_inn 1000 \
--mrce_max_itr 100 \
--mrce_TOL_type 1 --mrce_TOL 1e-3 \
--mrce_step_type 2 --mrce_const_ss 0.05 \
# --inner_cvx_solver 
# --cscc_pMat
# --no_constraints  --cscc_outer_verbose  --mrce_verbose \
# --cscc_plot_in_loop --mrce_verbose_plots