import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle, yaml, os, sys

with open('../config.yaml') as info:
    info_dict = yaml.load(info)

def check_single_edge(i, j, sdata, fdata):

    if sdata.keys() != fdata.keys():
        print('Subject list mismatch between sturctural and functional.')
        exit()

    s_edge = [sdata[sub][i,j] for sub in sdata.keys()]
    f_edge = [fdata[sub][i,j] for sub in fdata.keys()]

    fit = np.polyfit(s_edge, f_edge, 1)
    fit_fn = np.poly1d(fit) 

    plt.plot(s_edge, f_edge, 'yo', s_edge, fit_fn(s_edge), '--k')
    # plt.xlim(0, 5)
    # plt.ylim(0, 12)

    return


def load_data(task):    
    
    with open(info_dict['data_dir']+info_dict['s_file'], 'rb') as f:
        sdata = pickle.load(f)
    with open(info_dict['data_dir']+\
        'corrmats_tfMRI_'+task+'_125mm_LR_ROI_scale33.p','rb') as f:
        fdata = pickle.load(f, encoding='latin1')
    return (sdata, fdata)

task = "LANGUAGE"
(sdata, fdata) = load_data(task)
i = 1; j = 2
check_single_edge(i, j, sdata, fdata)









