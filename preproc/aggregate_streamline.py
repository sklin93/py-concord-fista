# TO AGGREGATE VALID STREAMLINE COUNTING MATRICES INTO A SIGNLE DATA STRUCTURE
# Example of csv file path: 
#       /work/code/data-STFN/structure_only/668361/Diffusion/fs125/668361_fs125_stmat.a.csv
#       /work/code/data-STFN/structure_only/668361/Diffusion/fs125/668361_fs125_stmat.q.csv
import numpy as np
import sys, os, re, pickle


def aggregate(root_dir, corrmat_file, sl_type):

    with open(corrmat_file, 'rb') as f:
        corrmat_dict = pickle.load(f)
    subject_list = corrmat_dict.keys()
    corrmat_filename, ext = os.path.splitext(os.path.split(corrmat_file)[1])

    slmat_dict = {}
    for sub_dir in os.listdir(root_dir):
        # check whether it is a subdir that is encoded as 6-digit subject id
        if os.path.isdir(os.path.join(root_dir,sub_dir)) and (sub_dir in subject_list): 
            slmat_dir = os.path.join(root_dir,sub_dir,'Diffusion','fs125')
            if os.path.isdir(slmat_dir):
                slmat_file = os.path.join(slmat_dir,sub_dir+'_fs125_stmat.'+sl_type+'.csv')
                if os.path.isfile(slmat_file):
                    print('Collecting streamline_counting_matrix: '+slmat_dir)
                    slmat = np.loadtxt(open(slmat_file))
                    slmat_dict[sub_dir] = slmat
                    print('Streamline_counting_matrix collected: '+slmat_dir)
                else:
                    print('Invalid streamline_counting_matrix file: '+slmat_file)
            else:
                print('Invalid streamline_counting_matrix dir: '+slmat_dir)
        else:
            print('Invalid subject dir: '+sub_dir)

    output_file = 'slmats.'+corrmat_filename+'.'+sl_type+'.p'
    with open(os.path.join(root_dir,output_file),'wb') as fp:
        pickle.dump(slmat_dict, fp, protocol=pickle.HIGHEST_PROTOCOL)
    return


if __name__ == '__main__':

    root_dir = sys.argv[1]      # /work/code/data-STFN/structure_only
    corrmat_file = sys.argv[2]  # /work/code/fs125/corrmats_tfMRI_WM_125mm_RL_ROI_scale33.p

    aggregate(root_dir, corrmat_file, 'a')
    aggregate(root_dir, corrmat_file, 'q')





