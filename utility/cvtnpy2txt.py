import sys
from hcp_cc import data_prep
import numpy as np

task = sys.argv[1]

vec_s, vec_f = data_prep(task, v1=False)
print(vec_s.shape)
print(vec_f.shape)

train_num = int(vec_s.shape[0] * 0.8)

np.savetxt('Xfile_'+task+'_train',vec_s[:train_num, :])
np.savetxt('Yfile_'+task+'_train',vec_f[:train_num, :])

np.savetxt('Xfile_'+task+'_test',vec_s[train_num:, :])
np.savetxt('Yfile_'+task+'_test',vec_f[train_num:, :])