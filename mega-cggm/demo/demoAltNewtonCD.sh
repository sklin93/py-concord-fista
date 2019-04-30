#!/bin/sh
(cd ../AltNewtonCD && make)
../AltNewtonCD/cggmfast_run -y 0.0272 -x 0.0004 167 3403 167 3403 data/Yfile_lang_train data/Xfile_lang_train Lambdafile_CD_lang_train_0.0272_0.0004 Thetafile_CD_lang_train_0.0272_0.0004 statsfile_CD_lang_train_0.0272_0.0004
# ../AltNewtonCD/cggmfast_run -y 0.1 -x 0.2 10 15 10 12 Yfile_demo Xfile_demo Lambdafile_demo Thetafile_demo statsfile_demo