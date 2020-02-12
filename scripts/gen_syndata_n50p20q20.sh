python cscc_merge.py \
--generate_synthetic \
--synthetic_dir 'data-utility/syndata_n50p20q20.pkl' \
--err_type 2 --pct_nnz 0.095 --base_nnz 0.9 \
--p 20 --q 20 --n 50 \
--pMat_noise 0.2  --success_prob_s1 0.15 --success_prob_s2 0.8 \
# --distr_type 2 --df 2