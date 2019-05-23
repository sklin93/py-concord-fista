#!/bin/sh
(cd ../AltNewtonBCD && make)
# ../AltNewtonBCD/hugecggm_run -y 0.1 -x 0.2 10 15 10 12 Yfile Xfile Lambdafile Thetafile statsfile
../AltNewtonBCD/hugecggm_run -y 0 -x 0 800 3403 800 3403 data/Yfile_syn_sf_train data/Xfile_syn_sf_train results/Lambdafile_BCD_syn_sf_train_0 results/Thetafile_BCD_syn_sf_train_0 results/statsfile_BCD_syn_sf_train_0