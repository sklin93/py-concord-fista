cd ../ && \
python cscc_fista.py --p_lambda 1 \
--synthetic_dir 'data-utility/syn.pkl' \
--outer_verbose --demo \
--TOL 1e-3 --TOL_inn 1e-5 \
--MAX_ITR 50 \
--step_type_out 3 --const_ss_out 0.05 \
--p_tau 0.7 \
--p_gamma 0.05 \
# --inner_cvx_solver --inner_verbose
# --no_constraints




