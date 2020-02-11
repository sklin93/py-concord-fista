#!/bin/bash
(cd ../AltNewtonCD && make)
res_dir='results2'
# for task in EMOTION #LANGUAGE MOTOR GAMBLING SOCIAL RELATIONAL WM
for task in EMOTION
do
	for i in {0..9}
	do
	# ../AltNewtonCD/cggmfast_run -y 0.1 -x 0.2 10 15 10 12 Yfile_demo Xfile_demo Lambdafile_demo Thetafile_demo statsfile_demo
	../AltNewtonCD/cggmfast_run -y 0.01 -x 0.001 -v 1 46 3403 46 3403 data/train_f_${task}_${i} data/train_s_${task}_${i} ${res_dir}/Lambdafile_${task}_${i} ${res_dir}/Thetafile_${task}_${i} ${res_dir}/statsfile_${task}_${i}
	
	done
done