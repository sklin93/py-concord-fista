cd ../ && \
python cscc_fista.py \
--generate_synthetic \
--synthetic_dir 'data-utility/syn_1000.pkl' \
--num_var 1000 \
--num_smp 500 \
--pct_nnz 0.05 \
--base_nnz 0.95 \
--overwrite