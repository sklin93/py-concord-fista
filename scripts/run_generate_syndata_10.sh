cd ../ && \
python cscc_fista.py \
--generate_synthetic \
--synthetic_dir 'data-utility/syndata_10.pkl' \
--num_var 10 \
--num_smp 50 \
--pct_nnz 0.2 \
--base_nnz 0.9 \
--overwrite