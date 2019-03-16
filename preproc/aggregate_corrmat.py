# TO AGGREGATE VALID CORRELATION MATRICES INTO A SIGNLE DATA STRUCTURE
import numpy as np
import sys, os, re, pickle

if __name__ == '__main__':

    root_dir = sys.argv[1]
    fmri_task = sys.argv[2]  # tfMRI_LANGUAGE_125mm_LR
    fmri_atlas = sys.argv[3] # ROI_scale33



    corrmat_dict = {}

    for sub_dir in os.listdir(root_dir):
        # check whether it is a subdir that is encoded as 6-digit subject id
        if os.path.isdir(os.path.join(root_dir,sub_dir)) and re.findall('[0-9]{6,6}', sub_dir): 
            corrmat_dir = os.path.join(root_dir,sub_dir,'timeseries',fmri_task,'Lausanne2008',fmri_atlas)
            if os.path.isdir(corrmat_dir):
                corrmat_file = os.path.join(corrmat_dir,'corrmat.fc')
                if os.path.isfile(corrmat_file):
                    print('Collecting corrmat: '+corrmat_dir)
                    corrmat = np.loadtxt(open(corrmat_file))
                    corrmat_dict[sub_dir] = corrmat
                    print('Corrmat collected: '+corrmat_dir)
                else:
                    print('Invalid corrmat file: '+corrmat_file)
            else:
                print('Invalid corrmat dir: '+corrmat_dir)
        else:
            print('Invalid subject dir: '+sub_dir)

    with open(os.path.join(root_dir,'aggregated_corrmat.p'), 'wb') as fp:
        pickle.dump(corrmat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)



