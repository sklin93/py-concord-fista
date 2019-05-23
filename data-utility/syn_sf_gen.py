import numpy as np
import pickle
import scipy.stats as st
from tqdm import tqdm
import sys
sys.path.append('../')
from hcp_cc import data_prep

def dist_mapping(name, params, n):
    if name == 'norm':
        return st.norm.rvs(*params, size=n)
    elif name == 'exponweib':
        return st.exponweib.rvs(*params, size=n)
    elif name == 'weibull_max':
        return st.weibull_max.rvs(*params, size=n)
    elif name == 'weibull_min':
        return st.weibull_min.rvs(*params, size=n)
    elif name == 'pareto':
        return st.pareto.rvs(*params, size=n)
    elif name == 'genextreme':
        return st.genextreme.rvs(*params, size=n)
    else:
        print(name, ' is invalid distribution.')

def get_best_distribution(data, n, verbose=False):
    dist_names = ["norm", "exponweib", "weibull_max", "weibull_min", "pareto", "genextreme"]
    dist_results = []
    params = {}
    for dist_name in dist_names:
        dist = getattr(st, dist_name)
        param = dist.fit(data)

        params[dist_name] = param
        # Applying the Kolmogorov-Smirnov test
        D, p = st.kstest(data, dist_name, args=param)
        if verbose:
            print("p value for "+dist_name+" = "+str(p))
        dist_results.append((dist_name, p))

    # select the best fitted distribution
    best_dist, best_p = (max(dist_results, key=lambda item: item[1]))
    # store the name of the best fit and its p value
    if verbose:
        print("Best fitting distribution: "+str(best_dist))
        print("Best p value: "+ str(best_p))
        print("Parameters for the best fit: "+ str(params[best_dist]))

    # return one col of S
    # import ipdb; ipdb.set_trace()
    return dist_mapping(best_dist, params[best_dist], n)

def gen_s_from_dist(vec_s, n):
    p = vec_s.shape[1]
    S = np.zeros((n, p))
    print('Generating S...')
    for i in tqdm(range(p)):
        S[:, i] = get_best_distribution(vec_s[:, i], n)
    print('S shape, min, max: ', S.shape, S.min(), S.max())
    return S

def gen_w(p):
    ''' generate p*p weight matrix'''
    # diagonal elements must be nonzero: created with ~N(0.8, 0.25)
    w = np.diag(np.random.normal(0.8, 0.5, p))
    ''' choose 3 neighbors for each S for corresponding F: 
        namely 3 nonzeros (apart from diag) for each col in w
        value are random between -1 and 1'''
    num_neighbor = 3
    for i in tqdm(range(p)):
        idx_ = np.random.choice(p, num_neighbor, replace=False)
        val_ = np.random.random(num_neighbor)*2-1
        for j in range(num_neighbor):
            # if it's diagonal entry, ignore
            if idx_[j] != i:
                w[idx_[j], i] = val_[j]
    print('Number of nonzeros in w: ', np.count_nonzero(w))
    print('Number of nonzeros in w diagonal: ', np.count_nonzero(w.diagonal()))
    print('Average degree (neighbors-only): ', (np.count_nonzero(w)-p)/p)
    return w

def gen_f_from_s(S):
    n, p = S.shape
    print('Generating F...')
    W = gen_w(p)
    F = S@W
    print('F shape: ', F.shape)
    return F, W

if __name__ == '__main__':
    SAMPLE_NUM = 10000
    vec_s, _ = data_prep('LANGUAGE', v1=False)
    S = gen_s_from_dist(vec_s, SAMPLE_NUM)
    F, W = gen_f_from_s(S)
    syn_data = {'S':S, 'F':F, 'W':W}
    with open('syn_sf.pkl', 'wb') as handle:
        pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)