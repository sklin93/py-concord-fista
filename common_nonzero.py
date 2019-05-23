"""Check pairwise overlap between two task's Omega_FS"""
import numpy as np
 
tasks = ['EMOTION','GAMBLING','LANGUAGE','MOTOR','RELATIONAL','SOCIAL','WM']
 
def load_omega(task,fdir='fs_results/',mid='_',lam=0.0009,vis=True):
    omega = np.load(fdir+str(lam)+mid+task+'.npy')
    omega = omega[:,omega.shape[0]:]
    if vis: 
        print(omega.shape)
        print(np.count_nonzero(omega))
    # omega[omega!=0]=1
    return omega
 
def nz_share(omega_i,omega_j):
    '''shared nonzero entry percentage, except diagonal'''
    d, _ = omega_i.shape
    nz_i = np.count_nonzero(omega_i) - d
    nz_j = np.count_nonzero(omega_j) - d
    shared = (nz_i+nz_j-np.count_nonzero(omega_i-omega_j))/2
    return shared/nz_i, shared/nz_j, shared
 
def edge_share():
    pass

def main():
    result = {}
    for i in range(len(tasks)):
        result[tasks[i]] = load_omega(tasks[i])
     
    ratio_mat = np.zeros((len(tasks),len(tasks)))
    for i in range(len(tasks)):
        for j in range(len(tasks)):
            omega_i = result[tasks[i]]
            omega_j = result[tasks[j]]
            import ipdb; ipdb.set_trace()
            ratio_i, ratio_j, _ = nz_share(omega_i,omega_j)
            ratio_mat[i,j] = round(ratio_i,5)
            ratio_mat[j,i] = round(ratio_j,5)
    print(ratio_mat)

if __name__ == '__main__':
    main()
    '''
    import pickle
    with open('data-utility/syn_sf.pkl', 'rb') as f:
        omega1 = pickle.load(f)['W']
        print(np.count_nonzero(omega1))
    omega2 = load_omega('syn_sf',mid='_train_',lam=0.00009)
    tmp1, tmp2, shared = nz_share(omega1, omega2)
    print(tmp1, tmp2, shared)
    '''