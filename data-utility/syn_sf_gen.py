import numpy as np
import pickle
import networkx as nx
import scipy.stats as st
from tqdm import tqdm, trange
import os, sys
os.chdir('../')
sys.path.append('./')
from hcp_cc import data_prep
os.chdir('./data-utility/')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

"""
'S' is sampled from HCP Effective-Resistance preprocessed structure distribution,
having the same variable dimension as HCP data.
"""
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
    return dist_mapping(best_dist, params[best_dist], n)

def gen_s_from_dist(vec_s, n, p_=None):
    '''n is sample size, p_ is variable number;
    if not specified, variable number will be the same as HCP data'''
    if p_ is None:
        p = vec_s.shape[1]
    else:
        p = p_
    S = np.zeros((n, p))
    print('Generating S...')
    for i in tqdm(range(p)):
        S[:, i] = get_best_distribution(vec_s[:, i], n)
    print('S shape, min, max: ', S.shape, S.min(), S.max())
    return S

"""
Generate synthetic struncture-function data (Random graph)
Each 'F_i' is a weighted sum of S_i and a fixed number of other regions. 
Namely, each node on the mapping graph has the same number of degree.
"""
def gen_w_rnd(p):
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

def gen_f_from_s_rnd(S):
    n, p = S.shape
    print('Generating F...')
    W = gen_w_rnd(p)
    F = S@W
    print('F shape: ', F.shape)
    return F, W

def gen_rnd():
    '''random network'''
    SAMPLE_NUM = 10000
    vec_s, _ = data_prep('LANGUAGE', v1=False)
    S = gen_s_from_dist(vec_s, SAMPLE_NUM)
    F, W = gen_f_from_s_rnd(S)
    syn_data = {'S':S, 'F':F, 'W':W}
    with open('syn_sf.pkl', 'wb') as handle:
        pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
Generate synthetic struncture-function data (Scale-free graph)
Mapping graph is a scale-free network. 
- Edge weights?
Each 'F_i' is gotten from random walk 1-5 steps starting from region i,
with transition probability based on network edge wights
- But then we don't have GT mapping?
- Also there's no need for S to follow HCP distribution, as the network hub of syn & gt will be different
"""

def sf_mapping(S):
    n, p = S.shape
    
    G = nx.barabasi_albert_graph(p, 5)
    # G = nx.scale_free_graph(p, alpha=0.13, beta=0.75, gamma=0.12)
    print(nx.info(G))
    A = np.asarray(nx.to_numpy_matrix(G))
    np.fill_diagonal(A, 1)
    '''
    import seaborn as sns
    deg = np.sum(A, axis=1)
    sns.distplot(deg, kde=False)
    plt.show()
    import powerlaw
    plt.figure()
    pl_fit = powerlaw.Fit(deg)
    fig = pl_fit.plot_cdf(linewidth=3, color='b')
    pl_fit.power_law.plot_cdf(ax=fig, color='g', linestyle='--')
    plt.show()
    '''
    # randomize edge weights
    for i in range(p):
        for j in range(i, p):
            if A[i, j] != 0:
                rnd_ = np.random.uniform(-1,1)
                A[i, j] = rnd_
                A[j, i] = rnd_
    return A

""" random walk
def gen_f_from_s_sf(S, A):
    n, p = S.shape
    '''Truncated normal distribution for generating Step Number: sn
       clip range: [1, 6], with mean=3, sd=1'''
    sn = list(map(int, np.round(st.truncnorm.rvs(a=-2, b=3, loc=3, size=p))))
    F = []
    for k in trange(n, desc='Subject loop'):
        cur_s = S[k, :]
        cur_f = np.zeros(p)
        for i in trange(p, desc='Parameter loop'):
            # for each S_i, random walk sn[i] steps
            cur_sn = sn[i]
            cur_pos = i
            cur_influencer = cur_s[cur_pos]
            cur_val = 0
            for step in range(cur_sn):
                if sum(A[cur_pos, :]) == 0:
                    # no neighbor, i.e. sum(A[cur_pos, :]) == 0
                    nxt_pos = cur_pos
                else:
                    # get the probabilities of walking to each node
                    prob = A[cur_pos, :]/sum(A[cur_pos, :])
                    # select next location based on the probabilities
                    nxt_pos = np.random.choice(np.arange(p), p=prob)
                cur_val += cur_influencer*w[cur_pos, nxt_pos]
                # update
                cur_pos = nxt_pos
                cur_influencer = cur_s[cur_pos]
            # how to define weight??
            # cur_f[cur_pos] += weight*cur_influencer
            # cur_f[cur_pos] += cur_influencer
        F.append(cur_f)
    import ipdb; ipdb.set_trace()
    return np.stack(F)
"""

def gen_sf(): 
    '''scale-free network'''
    SAMPLE_NUM = 10000
    if os.path.isfile('syn_sf_rnd.pkl'):
        with open('syn_sf_rnd.pkl', 'rb') as f:
            S = pickle.load(f)['S']
            if SAMPLE_NUM < S.shape[0]:
                S = S[:SAMPLE_NUM, :]
    else:
        vec_s, _ = data_prep('LANGUAGE', v1=False)
        S = gen_s_from_dist(vec_s, SAMPLE_NUM)

    # Shuffling variable to see if the artifacts still exists at the same column
    # Permutation matrix
    P = np.eye(S.shape[1])
    np.random.shuffle(P)
    S = S@P
    print('S generated. Shape: ', S.shape)

    W = sf_mapping(S)
    print('Scale-free mapping network generated.')

    F = S@W
    print('F shape: ', F.shape)
    # syn_data = {'S':S, 'F':F, 'W':W}
    syn_data = {'S':S, 'F':F, 'W':W, 'P':P}

    with open('syn_sf_sf_shuffled.pkl', 'wb') as handle:
        pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    gen_sf()