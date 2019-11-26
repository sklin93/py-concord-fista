cd ../ && \
python cscc_fista.py --p_lambda 1 \
--synthetic_dir 'data-utility/syn.pkl' \
--outer_verbose --inner_verbose --demo \
--TOL 1e-3 --TOL_inn 1e-3 \
--MAX_ITR 50 \
--step_type_out 3 --const_ss_out 0.05 \
--p_tau 0.7 \
--p_gamma 0.05 \
--inner_cvx_solver
# --no_constraints

# Inferred Omega:
# [[ 1.05  0.00  0.00  0.00  0.00  0.00  0.65]
#  [ 0.00  0.94  0.00  0.61  0.64  0.87  0.00]
#  [ 0.00  0.00  1.00  0.00  0.00  0.00  0.00]
#  [ 0.00  0.61  0.00  1.25  0.45  0.58  0.00]
#  [ 0.00  0.64  0.00  0.45  1.30  0.75  0.00]
#  [ 0.00  0.87  0.00  0.58  0.75  1.50  0.00]
#  [ 0.65  0.00  0.00  0.00  0.00  0.00  1.34]]
