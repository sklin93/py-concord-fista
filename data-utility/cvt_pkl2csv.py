import pickle
import numpy as np

with open('syn.pkl','rb') as f:
	syn = pickle.load(f)

import ipdb; ipdb.set_trace()
data = syn[2]

np.savetxt("syn_data.csv", data, delimiter=",")