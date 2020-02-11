# for j in 0.02511886 0.05011872 0.19952623 0.39810717
# for j in 0.01 0.01467799
# for j in 0.005275 0.00879923
# do
# 	python hcp_mrce.py EMOTION 0.1 ${j}
# 	python hcp_mrce.py LANGUAGE 0.1 ${j}
# 	# python hcp_mrce.py MOTOR 0.1 ${j}
# 	# python hcp_mrce.py GAMBLING 0.1 ${j}
# 	# python hcp_mrce.py SOCIAL 0.1 ${j}
# 	# python hcp_mrce.py RELATIONAL 0.1 ${j}
# 	# python hcp_mrce.py WM 0.1 ${j}
# done

for i in 0.02511886 0.05011872 0.19952623 0.39810717
do
	for j in 0.05011872
	do
		# python hcp_mrce.py EMOTION ${i} ${j}
		# python hcp_mrce.py LANGUAGE ${i} ${j}
		python hcp_mrce.py MOTOR ${i} ${j}
		# python hcp_mrce.py GAMBLING ${i} ${j}
		# python hcp_mrce.py SOCIAL ${i} ${j}
		python hcp_mrce.py RELATIONAL ${i} ${j}
		# python hcp_mrce.py WM ${i} ${j}
	done
done