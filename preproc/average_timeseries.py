import numpy as np
import sys, glob, os

# For each ROI, returns a row of voxelwise averaged timeseries.
# All zero reads will be removed.

# Voxel-level timeseries of ROI "i" are stored as "i.txt" under the directory "dir_ts".
# There should not be any other ".txt" files under the same directory "dir_ts".

if __name__ == '__main__':
    dir_ts      = sys.argv[1]  
    output_file = sys.argv[2]

    # load the first timeseries file to extract dimension information
    ts_raw      = np.loadtxt(open(dir_ts+'/1.txt', "rb"), delimiter=" ")
    ts_list     = os.listdir(dir_ts)
    ts_list     = [i for i in ts_list if i.endswith('.txt')]
    num_frame   = ts_raw.shape[1] - 6 # remove coordinate prefix
    num_roi     = len(ts_list)

    ts_mean = np.empty([num_roi, num_frame])
    for i in range(num_roi):
        ts_raw     = np.loadtxt(open(dir_ts+'/'+str(i+1)+'.txt', "rb"), delimiter=" ")
        nzn_voxel  = np.nonzero(np.sum(ts_raw[:,6:], axis=1))
        ts_mean[i] = np.mean(ts_raw[nzn_voxel[0],6:], axis=0) 

    np.savetxt(dir_ts+'/'+output_file, ts_mean, delimiter=" ", fmt='%0.3f') 