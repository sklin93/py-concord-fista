import numpy as np
import pickle
import networkx as nx
import scipy.stats as st
from numpy import linalg as LA
from tqdm import tqdm, trange
import os, sys
os.chdir('../')
sys.path.append('./')
from hcp_cc import data_prep
os.chdir('./data-utility/')
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def gen_s_from_dist(vec_s, n, p_=None):
    '''
    'S' is sampled from HCP Effective-Resistance preprocessed structure distribution,
    having the same variable dimension as HCP data.
    n is sample size, p_ is the desired variable number;
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
Generate synthetic struncture-function data (mapping is Random graph)
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
Generate synthetic struncture-function data (mapping is scale-free graph)
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

def gen_sf(noise_type=0, shuffle=False): 
    '''scale-free network
    noise_type: 0 being no noise, 
    1 being independent Gaussian,
    2 being multivarite Gaussian with Sigma,
    3 being idependent t-distribution,
    4 being multivariate t-distribution
    '''

    SAMPLE_NUM = 10000
    if os.path.isfile('syn_sf_rnd.pkl'):
        with open('syn_sf_rnd.pkl', 'rb') as f:
            S = pickle.load(f)['S']
            if SAMPLE_NUM < S.shape[0]:
                S = S[:SAMPLE_NUM, :]
    else:
        vec_s, _ = data_prep('LANGUAGE', v1=False)
        S = gen_s_from_dist(vec_s, SAMPLE_NUM)

    p = S.shape[1]

    if shuffle:
        '''Shuffling variables to see if the artifacts still exists at the same column'''
        P_ = np.eye(p)  # permutation matrix
        np.random.shuffle(P_)
        S = S@P_     # shuffle columns
    
    print('S generated. Shape: ', S.shape)

    W = sf_mapping(S)
    print('Scale-free mapping network generated.')

    F = S@W
    if noise_type == 1:
        '''noise sampled from ~N(0, 1/3 current F's sd)'''
        SNR = []
        for i in tqdm(range(p)):
            cur_noise = np.random.normal(0, np.std(F[:, i])/3.0, SAMPLE_NUM)
            SNR.append(np.var(F[:,i])/np.var(cur_noise))
            F[:, i] += cur_noise
        print(np.mean(SNR))
    if noise_type == 2:
        '''noise sampled from ~N(0, Cov), with Cov's diag being the same as in noise_type 1'''
        # initialize random covariance
        cov_range = 1e-5
        cov = np.random.uniform(-cov_range, cov_range, p*p).reshape(p, p)
        cov = cov @ cov.T
        for i in tqdm(range(p)):
            cov[i, i] = np.std(F[:, i])/1.0e3
        cov = nearestPD(cov)
        noise = np.random.multivariate_normal(np.zeros(p), cov, SAMPLE_NUM)
        # signal-to-noise ratio
        print(np.mean(np.var(F,axis=0)/ np.var(noise,axis=0)))
        F += noise
    if noise_type == 3:
        '''noise sampled from ~T(df=5, 0, 1/3 current F's sd)'''
        df = 5
        for i in tqdm(range(p)):
            cur_noise = st.t.rvs(df, loc=0, scale=np.std(F[:, i])/3.0, size=SAMPLE_NUM)
            F[:, i] += cur_noise
    if noise_type == 4:
        '''noise sampled from T(df=5, 0, Cov), Cov is the same as noise_type 2'''
        cov = np.random.uniform(-0.001, 0.001, p*p).reshape(p, p)
        cov = cov @ cov.T
        for i in tqdm(range(p)):
            cov[i, i] = np.std(F[:, i])/3.0
        cov = nearestPD(cov)
        noise = multivariate_t_rvs(np.zeros(p), cov, df=5, n=SAMPLE_NUM)
        F += noise
    print('F shape: ', F.shape)

    # saving data
    if shuffle:
        syn_data = {'S':S, 'F':F, 'W':W, 'P':P_}
        with open('syn_sf_sf_'+str(noise_type)+'_shuffled.pkl', 'wb') as handle:
            pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    else:
        syn_data = {'S':S, 'F':F, 'W':W}
        with open('syn_sf_sf_'+str(noise_type)+'.pkl', 'wb') as handle:
            pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
Generation methods from CGGM(and other papers)
"""

def gen_invcov(p, thresh = 0.997):
    '''Lambda matrix in CGGM. Method found in paper:
    An inexact interior point method for L1-regularized sparse covariance selection'''
    U = np.random.rand(p,p)*2-1
    U[np.where(np.logical_and(U>=-thresh, U<=thresh))] = 0
    U[np.where(U>thresh)] = 1
    U[np.where(U<-thresh)] = -1
    print(np.count_nonzero(U), np.count_nonzero(U)/(p*p))
    # print(set(U.flatten()))
    A = U.T @ U
    d = A.diagonal()
    A = np.maximum(np.minimum(A - np.diag(d), 1), -1)
    A = A + np.diag(d + 1)
    invcov = A + max(-1.2 * min(LA.eigvals(A)), 1e-4) * np.eye(p)
    return invcov

def gen_theta(p, q):
    '''Theta matrix in CGGM. Data: X: p-dimensional, Y: q-dimensional''' 
    a = 10 # 100 in CGGM
    b = 3 # 10 in CGGM
    num_influencer = min(p, int(a * np.sqrt(p))) # which x has influence on y
    num_edges = b * q # total number of edges across x and y
    theta_ = np.zeros(num_influencer * q)
    theta_[np.random.choice(num_influencer*q, num_edges, replace=False)] = 1
    theta_ = theta_.reshape(num_influencer, q)
    theta = np.zeros((p,q))
    chosen_influencer = np.random.choice(p, num_influencer, replace=False)
    for i in chosen_influencer:
        theta[i, :] = theta_[0]
        theta_ = np.delete(theta_, 0, 0)
    return theta

def gen_CGGM():
    SAMPLE_NUM = 1000
    p = 1000
    '''
    # Using real data's structural distribution
    if os.path.isfile('syn_sf_rnd.pkl'):
        with open('syn_sf_rnd.pkl', 'rb') as f:
            S = pickle.load(f)['S']
            if SAMPLE_NUM < S.shape[0]:
                S = S[:SAMPLE_NUM, :]
            assert p <= S.shape[1], 'cannot select more dimensions\
                                     than the variable dimensions'
            S = S[:, :p]
    else:
        vec_s, _ = data_prep('LANGUAGE', v1=False)
        S = gen_s_from_dist(vec_s, SAMPLE_NUM)
        assert p <= S.shape[1], 'cannot select more dimensions\
                                 than the variable dimensions'
        S = S[:, :p]
    '''
    # Sample S from multivariate normal distribution
    # S entries are more connected to each other, set thresh smaller (more nonzeros)
    S = np.random.multivariate_normal(np.zeros(p), 
                LA.inv(gen_invcov(p, thresh=0.5)), SAMPLE_NUM)
    print('S shape: ', S.shape)
    
    Theta = gen_theta(p, p)
    Lambda = gen_invcov(p)
    cov = LA.inv(Lambda)
    B = - Theta @ cov
    F = []
    for i in tqdm(range(SAMPLE_NUM)):
        mean = B.T @ S[i]
        F.append(np.random.multivariate_normal(mean, cov))
    F = np.stack(F)
    syn_data = {'S':S, 'F':F, 'Theta':Theta, 'Lambda':Lambda}
    with open('syn_sf_CGGM.pkl', 'wb') as handle:
        pickle.dump(syn_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    gen_sf(noise_type=1, shuffle=False)
    # gen_CGGM()