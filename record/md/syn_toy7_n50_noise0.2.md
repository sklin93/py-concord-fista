
python cscc_merge.py --generate_synthetic --synthetic_dir 'data-utility/syn_pMat0.2_n50.pkl' --err_type 2 --p 7 --q 7 --n 50 --pMat_noise 0.2


= = = iteration 19 = = = 

 = = = line-search iteration 0 = = = 

- - - OUTER problem solution UPDATED - - -
1st(diag) term: 2.239225 | 2nd(trace) term: 7.000707 | 3rd(penalty) term: 0.002044
error: 3.13, subg norm:7.09
h function value (data fidelity):9.239932
h function comparable value:4.61997
f function value:9.24198
Inferred Omega:
[[ 9.24e-01  0.00e+00  0.00e+00  0.00e+00 -1.68e-04 -1.38e-04  0.00e+00]
 [ 0.00e+00  8.35e-01  0.00e+00  0.00e+00  0.00e+00  0.00e+00 -1.89e-04]
 [ 0.00e+00  0.00e+00  7.90e-01  0.00e+00 -8.96e-05  0.00e+00 -1.54e-04]
 [ 0.00e+00  0.00e+00  0.00e+00  9.18e-01  0.00e+00  0.00e+00  0.00e+00]
 [-1.68e-04  0.00e+00 -8.96e-05  0.00e+00  8.08e-01 -1.21e-04 -9.00e-05]
 [-1.38e-04  0.00e+00  0.00e+00  0.00e+00 -1.21e-04  8.53e-01  7.31e-05]
 [ 0.00e+00 -1.89e-04 -1.54e-04  0.00e+00 -9.00e-05  7.31e-05  8.45e-01]]
nonzero entry count:  23
symmetric(Th, G_n, Omg_n):True,True,True
dumping records:syn_pMat0.2_n50_lg(1.0,0.7)_ITR(20,100)_step(1,0.7)


= = = Finished = = =
Groundtruth Omega:
[[ 1.00e+00  0.00e+00  0.00e+00  0.00e+00  3.76e-01  2.85e-01  0.00e+00]
 [ 0.00e+00  1.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  3.64e-01]
 [ 0.00e+00  0.00e+00  1.00e+00  0.00e+00  3.06e-01  0.00e+00  3.24e-01]
 [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00  0.00e+00  0.00e+00  0.00e+00]
 [ 3.76e-01  0.00e+00  3.06e-01  0.00e+00  1.00e+00  2.76e-01  2.55e-01]
 [ 2.85e-01  0.00e+00  0.00e+00  0.00e+00  2.76e-01  1.00e+00  0.00e+00]
 [ 0.00e+00  3.64e-01  3.24e-01  0.00e+00  2.55e-01  0.00e+00  1.00e+00]]
nonzero entry count:  21
Inferred Omega:
[[ 9.24e-01  0.00e+00  0.00e+00  0.00e+00 -1.68e-04 -1.38e-04  0.00e+00]
 [ 0.00e+00  8.35e-01  0.00e+00  0.00e+00  0.00e+00  0.00e+00 -1.89e-04]
 [ 0.00e+00  0.00e+00  7.90e-01  0.00e+00 -8.96e-05  0.00e+00 -1.54e-04]
 [ 0.00e+00  0.00e+00  0.00e+00  9.18e-01  0.00e+00  0.00e+00  0.00e+00]
 [-1.68e-04  0.00e+00 -8.96e-05  0.00e+00  8.08e-01 -1.21e-04 -9.00e-05]
 [-1.38e-04  0.00e+00  0.00e+00  0.00e+00 -1.21e-04  8.53e-01  7.31e-05]
 [ 0.00e+00 -1.89e-04 -1.54e-04  0.00e+00 -9.00e-05  7.31e-05  8.45e-01]]
nonzero entry count: 23







 = = = iteration 19 = = = 

 = = = line-search iteration 0 = = = 
~ ~ ~ No constraint is applied.

- - - OUTER problem solution UPDATED - - -
1st(diag) term: 2.238519 | 2nd(trace) term: 7.000000 | 3rd(penalty) term: 0.000000
error: 1.17, subg norm:2.65
h function value (data fidelity):9.238519
h function comparable value:4.61926
f function value:9.23852
Inferred Omega:
[[ 9.24e-01  0.00e+00 -0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00]
 [ 0.00e+00  8.35e-01 -0.00e+00 -0.00e+00 -0.00e+00 -0.00e+00  0.00e+00]
 [-0.00e+00 -0.00e+00  7.90e-01  0.00e+00  0.00e+00 -0.00e+00  0.00e+00]
 [ 0.00e+00 -0.00e+00  0.00e+00  9.18e-01 -0.00e+00  0.00e+00 -0.00e+00]
 [ 0.00e+00 -0.00e+00  0.00e+00 -0.00e+00  8.08e-01  0.00e+00  0.00e+00]
 [ 0.00e+00 -0.00e+00 -0.00e+00  0.00e+00  0.00e+00  8.53e-01 -0.00e+00]
 [ 0.00e+00  0.00e+00  0.00e+00 -0.00e+00  0.00e+00 -0.00e+00  8.45e-01]]
nonzero entry count:  7
symmetric(Th, G_n, Omg_n):True,True,True
dumping records:syn_pMat0.2_n50_unconstrained_lg(1.0,0.7)_ITR(20,100)_step(1,0.7)


= = = Finished = = =
Groundtruth Omega:
[[ 1.00e+00  0.00e+00  0.00e+00  0.00e+00  3.76e-01  2.85e-01  0.00e+00]
 [ 0.00e+00  1.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00  3.64e-01]
 [ 0.00e+00  0.00e+00  1.00e+00  0.00e+00  3.06e-01  0.00e+00  3.24e-01]
 [ 0.00e+00  0.00e+00  0.00e+00  1.00e+00  0.00e+00  0.00e+00  0.00e+00]
 [ 3.76e-01  0.00e+00  3.06e-01  0.00e+00  1.00e+00  2.76e-01  2.55e-01]
 [ 2.85e-01  0.00e+00  0.00e+00  0.00e+00  2.76e-01  1.00e+00  0.00e+00]
 [ 0.00e+00  3.64e-01  3.24e-01  0.00e+00  2.55e-01  0.00e+00  1.00e+00]]
nonzero entry count:  21
Inferred Omega:
[[ 9.24e-01  0.00e+00 -0.00e+00  0.00e+00  0.00e+00  0.00e+00  0.00e+00]
 [ 0.00e+00  8.35e-01 -0.00e+00 -0.00e+00 -0.00e+00 -0.00e+00  0.00e+00]
 [-0.00e+00 -0.00e+00  7.90e-01  0.00e+00  0.00e+00 -0.00e+00  0.00e+00]
 [ 0.00e+00 -0.00e+00  0.00e+00  9.18e-01 -0.00e+00  0.00e+00 -0.00e+00]
 [ 0.00e+00 -0.00e+00  0.00e+00 -0.00e+00  8.08e-01  0.00e+00  0.00e+00]
 [ 0.00e+00 -0.00e+00 -0.00e+00  0.00e+00  0.00e+00  8.53e-01 -0.00e+00]
 [ 0.00e+00  0.00e+00  0.00e+00 -0.00e+00  0.00e+00 -0.00e+00  8.45e-01]]
nonzero entry count: 7
