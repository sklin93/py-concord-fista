import numpy as np
import sys, os

# Compute the correlation matrix given ROI-level timeseries.
# ROI-level timeseries are given via "ts_file" under the directory "dir_ts".

if __name__ == '__main__':
    dir_ts       = sys.argv[1]  
    ts_file      = sys.argv[2]
    corrmat_file = sys.argv[3]

    ts = np.loadtxt(open(dir_ts+'/'+ts_file, "rb"), delimiter=" ")
    num_roi, num_frame = ts.shape
    corr_mat = np.corrcoef(ts)
    np.savetxt(dir_ts+'/'+corrmat_file, corr_mat, delimiter=" ", fmt='%0.5f') 