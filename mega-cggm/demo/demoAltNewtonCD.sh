#!/bin/sh
(cd ../AltNewtonCD && make)
../AltNewtonCD/cggmfast_run -y 0.000001 -x 0.000001 800 3403 800 3403 data/Yfile_syn_sf_train data/Xfile_syn_sf_train Lambdafile_CD_syn_sf_train_0.000001 Thetafile_CD_syn_sf_train_0.000001 statsfile_CD_syn_sf_train_0.000001
# ../AltNewtonCD/cggmfast_run -y 0.1 -x 0.2 10 15 10 12 Yfile_demo Xfile_demo Lambdafile_demo Thetafile_demo statsfile_demo