#!/bin/sh
(cd ../Pseudo && make)
../Pseudo/pseudo_run -v 1 -y 0.2 -x 0.2 30 3403 30 3403 Yfile Xfile Lambdafile Thetafile statsfile
