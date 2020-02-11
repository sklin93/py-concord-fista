cd ../ && \
python cscc_fista.py --p_lambda 0.1 \
--outer_verbose --inner_verbose \
--MAX_ITR 10 --step_type_out 1 --const_ss_out 0.05 --TOL 1e-3  \
--p_tau 0.1 --p_gamma 0.1 \
--MAX_ITR_inn 50000 --step_type_inn 1 --const_ss_inn 0.05 --TOL_inn 1e-20  \
--p_kappa 0.5 \
--synthetic_dir 'data-utility/syn.pkl' \
--demo #--inner_cvx_solver #--plot_past_records