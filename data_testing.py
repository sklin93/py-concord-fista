import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

import sys
import numpy as np
import pickle
import scipy.stats as st
import seaborn as sns

from hcp_cc import data_prep

'''Normal test'''
# task = sys.argv[1]
# vec_s, _ = data_prep(task, v1=False)
# _, pval = st.normaltest(vec_s)
# not_normal = np.where(pval < 1e-03)[0]
# print('There are ', len(not_normal), ' edges not being Gaussian: ', not_normal)

'''S edge weights test'''
# vec_s, _ = data_prep('LANGUAGE', v1=False)
# print(vec_s.shape)

# # top 10 dense columns
# fig, axs = plt.subplots(2,5,figsize=(15, 6))
# axs = axs.ravel()
# top10 = [2623, 1863, 1866, 1864, 5, 26, 1875, 1868, 1871, 1917]
# for i in range(10):
# 	sns.distplot(vec_s[:, top10[i]], kde=True, ax=axs[i])
# plt.show()
# # rnd middle 5 columns
# fig, axs = plt.subplots(2,5,figsize=(15, 6))
# axs = axs.ravel()
# mid10 = [660,	3090, 1341, 1073,  748, 2915, 2047, 1727, 1771,  867]
# for i in range(10):
# 	sns.distplot(vec_s[:, mid10[i]], kde=True, ax=axs[i])
# plt.show()		
# # least 10
# fig, axs = plt.subplots(2,5,figsize=(15, 6))
# axs = axs.ravel()
# last10 = [1385, 1398, 1400, 3239, 2227, 2224, 2222, 2219, 2208, 2317]
# for i in range(10):
# 	sns.distplot(vec_s[:, last10[i]], kde=True, ax=axs[i])
# plt.show()

'''Per-node basis test'''
task = sys.argv[1]
vec_s, _ = data_prep(task, v1=False, flatten=False)
print(vec_s.shape)
# average over all subjects
avg_s = np.mean(vec_s, axis=2)
#choose random 10 nodes to plot their edge weights distributions
fig, axs = plt.subplots(2,5,figsize=(15, 6))
axs = axs.ravel()
idx = np.random.choice(np.arange(avg_s.shape[0]), 10, replace=False)
print(idx)
for i in range(10):
	sns.distplot(avg_s[idx[i]], kde=True, ax=axs[i])
plt.show()