#!/bin/sh
(cd ../AltNewtonCD && make)
../AltNewtonCD/cggmfast_run -y 0.005 -x 0.005 30 4005 30 4005 Yfile Xfile Lambdafile_CD Thetafile_CD statsfile_CD
# ../AltNewtonCD/cggmfast_run -y 0.1 -x 0.2 10 15 10 12 Yfile_demo Xfile_demo Lambdafile_demo Thetafile_demo statsfile_demo