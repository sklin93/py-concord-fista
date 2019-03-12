#!/bin/sh
(cd ../AltNewtonBCD && make)
# ../AltNewtonBCD/hugecggm_run -y 0.1 -x 0.2 10 15 10 12 Yfile Xfile Lambdafile Thetafile statsfile
../AltNewtonBCD/hugecggm_run -y 0.02 -x 0.02 30 4005 30 4005 Yfile Xfile Lambdafile_BCD Thetafile_BCD statsfile_BCD