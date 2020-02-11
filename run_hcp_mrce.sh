# for i in 0.02511886 0.05011872 0.1        0.19952623 0.39810717
# do
# 	for j in 0.02511886 0.05011872 0.1        0.19952623 0.39810717
# 	do
# 		nohup python -u hcp_mrce.py EMOTION ${i} ${j} > logs/EMOTION_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py LANGUAGE ${i} ${j} > logs/LANGUAGE_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py MOTOR ${i} ${j} > logs/MOTOR_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py GAMBLING ${i} ${j} > logs/GAMBLING_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py SOCIAL ${i} ${j} > logs/SOCIAL_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py RELATIONAL ${i} ${j} > logs/RELATIONAL_cscc_mrce_${i}_${j} 2>&1 &
# 		nohup python -u hcp_mrce.py WM ${i} ${j} > logs/WM_cscc_mrce_${i}_${j} 2>&1 &
# 	done
# done

# for i in 0.02511886 0.05011872 0.1        0.19952623 0.39810717
# do
# 	# nohup python -u hcp_mrce.py EMOTION ${i} 0.1 > logs/EMOTION_cscc_mrce_${i}_0.1 2>&1 &
# 	# nohup python -u hcp_mrce.py LANGUAGE ${i} 0.1 > logs/LANGUAGE_cscc_mrce_${i}_0.1 2>&1 &
# 	# nohup python -u hcp_mrce.py MOTOR ${i} 0.1 > logs/MOTOR_cscc_mrce_${i}_0.1 2>&1 &
# 	# nohup python -u hcp_mrce.py GAMBLING ${i} 0.1 > logs/GAMBLING_cscc_mrce_${i}_0.1 2>&1 &
# 	# nohup python -u hcp_mrce.py SOCIAL ${i} 0.1 > logs/SOCIAL_cscc_mrce_${i}_0.1 2>&1 &
# 	# nohup python -u hcp_mrce.py RELATIONAL ${i} 0.1 > logs/RELATIONAL_cscc_mrce_${i}_0.1 2>&1 &
# 	nohup python -u hcp_mrce.py WM ${i} 0.1 > logs/WM_cscc_mrce_${i}_0.1 2>&1 &
# done

for i in 0.05011872 0.1 #0.02511886 0.05011872 0.1 0.19952623 0.39810717
do
	for j in 0.39810717
	do
		nohup python -u hcp_mrce.py EMOTION ${i} ${j} > logs/EMOTION_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py LANGUAGE ${i} ${j} > logs/LANGUAGE_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py MOTOR ${i} ${j} > logs/MOTOR_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py GAMBLING ${i} ${j} > logs/GAMBLING_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py SOCIAL ${i} ${j} > logs/SOCIAL_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py RELATIONAL ${i} ${j} > logs/RELATIONAL_cscc_mrce_${i}_${j} 2>&1 &
		# nohup python -u hcp_mrce.py WM ${i} ${j} > logs/WM_cscc_mrce_${i}_${j} 2>&1 &
	done
done

# # for j in 0.02511886 0.05011872 0.19952623 0.39810717
# # for j in 0.01 0.01467799
# for j in 0.001 0.00316228 #0.005275 0.00879923
# do
# 	nohup python -u hcp_mrce.py EMOTION 0.1 ${j} > logs/EMOTION_cscc_mrce_0.1_${j} 2>&1 &
# 	nohup python -u hcp_mrce.py LANGUAGE 0.1 ${j} > logs/LANGUAGE_cscc_mrce_0.1_${j} 2>&1 &
# 	# nohup python -u hcp_mrce.py MOTOR 0.1 ${j} > logs/MOTOR_cscc_mrce_0.1_${j} 2>&1 &
# 	# nohup python -u hcp_mrce.py GAMBLING 0.1 ${j} > logs/GAMBLING_cscc_mrce_0.1_${j} 2>&1 &
# 	# nohup python -u hcp_mrce.py SOCIAL 0.1 ${j} > logs/SOCIAL_cscc_mrce_0.1_${j} 2>&1 &
# 	# nohup python -u hcp_mrce.py RELATIONAL 0.1 ${j} > logs/RELATIONAL_cscc_mrce_0.1_${j} 2>&1 &
# 	# nohup python -u hcp_mrce.py WM 0.1 ${j} > logs/WM_cscc_mrce_0.1_${j} 2>&1 &
# done