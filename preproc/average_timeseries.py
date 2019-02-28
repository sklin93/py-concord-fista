import numpy as np
import sys, glob


# for each ROI, returns a row of voxelwise averaged timeseries
def average_ts(ts_clean):
    return ts_mean

# for each ROI, returns a row of voxelwise cleaned timeseries
# all zero reads will be removed
def remove_zeros(ts_raw):
    return ts_clean


if __name__ == '__main__':
    dir_ts      = sys.argv[1]
    output_file = sys.argv[2]


    num_roi
    num_frame

    ts_mean = np.empty(num_roi, num_frame)

    for file_path in glob.iglob(dir_ts + '/*.txt'):
        ts = np.loadtxt(open(file_path, "rb"), delimiter=",")
        ts_mean = average_ts(remove_zeros(ts))

    np.savetxt(output_file, ts_mean, delimiter=",") 


