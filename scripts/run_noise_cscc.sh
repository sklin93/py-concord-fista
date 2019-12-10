cd ../ && \
python cscc_merge.py \
--synthetic_dir 'data-utility/syn_pMat0.2_n500_p1000.2.pkl' \
--run_cscc --verbose \
--cscc_lambda 0.1 \
--cscc_outer_verbose \
--cscc_TOL 1e-3 --cscc_TOL_inn 1e-2 \
--cscc_max_itr 100 \
--cscc_step_type_out 1 --cscc_const_ss_out 0.1 \
--cscc_tau 0.7 \
--cscc_gamma 0.1 \
# --no_constraints
# --inner_cvx_solver\