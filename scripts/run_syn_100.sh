cd ../ && \
python cscc_fista.py \
--outer_verbose \
--MAX_ITR 100 \
--step_type_out 3  \
--const_ss_out 0.15 \
--p_tau 0.2 \
--p_gamma 0.1 --p_lambda 0.2 \
--TOL 1e-3 --TOL_inn 1e-3 \
--synthetic_dir 'data-utility/syn_100.pkl' \
--demo #--no_constraints \
# --plot_past_records