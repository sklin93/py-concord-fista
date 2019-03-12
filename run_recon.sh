# without train-val split
nohup python -u f_recon.py EMOTION 0.33 > emotion_hc 2>&1 &
nohup python -u f_recon.py GAMBLING 0.28 > gambling_hc 2>&1 &
nohup python -u f_recon.py LANGUAGE 0.36 > language_hc 2>&1 &
nohup python -u f_recon.py MOTOR 0.36 > motor_hc 2>&1 &
nohup python -u f_recon.py RELATIONAL 0.09 > relational_hc 2>&1 &
nohup python -u f_recon.py SOCIAL 0.17 > social_hc 2>&1 &
nohup python -u f_recon.py WM 0.28 > wm_hc 2>&1 &

# with train-val split 
# nohup python -u f_recon.py EMOTION 0.33 > emotion_train_hc 2>&1 &
# nohup python -u f_recon.py GAMBLING 0.28 > gambling_train_hc 2>&1 &
# nohup python -u f_recon.py LANGUAGE 0.36 > language_train_hc 2>&1 &
# nohup python -u f_recon.py MOTOR 0.36 > motor_train_hc 2>&1 &
# nohup python -u f_recon.py RELATIONAL 0.09 > relational_train_hc 2>&1 &
# nohup python -u f_recon.py SOCIAL 0.17 > social_train_hc 2>&1 &
# nohup python -u f_recon.py WM 0.28 > wm_train_hc 2>&1 &

# nohup python -u f_recon.py EMOTION 0.33 > emotion_train_rnd1 2>&1 &
# nohup python -u f_recon.py GAMBLING 0.28 > gambling_train_rnd1 2>&1 &
# nohup python -u f_recon.py LANGUAGE 0.36 > language_train_rnd1 2>&1 &
# nohup python -u f_recon.py MOTOR 0.36 > motor_train_rnd1 2>&1 &
# nohup python -u f_recon.py RELATIONAL 0.09 > relational_train_rnd1 2>&1 &
# nohup python -u f_recon.py SOCIAL 0.17 > social_train_rnd1 2>&1 &
# nohup python -u f_recon.py WM 0.28 > wm_train_rnd1 2>&1 &

# nohup python -u f_recon.py EMOTION 0.33 > emotion_train_rnd2 2>&1 &
# nohup python -u f_recon.py GAMBLING 0.28 > gambling_train_rnd2 2>&1 &
# nohup python -u f_recon.py LANGUAGE 0.36 > language_train_rnd2 2>&1 &
# nohup python -u f_recon.py MOTOR 0.36 > motor_train_rnd2 2>&1 &
# nohup python -u f_recon.py RELATIONAL 0.09 > relational_train_rnd2 2>&1 &
# nohup python -u f_recon.py SOCIAL 0.17 > social_train_rnd2 2>&1 &
# nohup python -u f_recon.py WM 0.28 > wm_train_rnd2 2>&1 &