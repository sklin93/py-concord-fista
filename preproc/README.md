# HCP Task-evoked fMRI preprocessing
- login to hcp-store.bic.ucsb.edu
- Data directory: /work/code/fs125 

## Preprocessing code: run ./preproc_tfmri.sh  under directory /work/gitrepo/py-concord-fista/preproc
- Grammar: ./preproc_tfmri.sh <path_to_data> <task_name> <atlas_name> <download_flag> <resampling_flag> <smoothing_flag> <timeseries_extraction_flag> <overwrite_flag>
- path_to_data: We use /work/code/fs125 without /  in the end. 
- Remember to write subject IDs into <path_to_data>/test_fs125_subject_list.txt before running the script. 
- <path_to_data>/full_subject_list.txt contains all subject IDs.
- task_name: Use all capital letters, LANGUAGE, WM, GAMBLING, SOCIAL, EMOTION, ...
- atlas_name: Options could be ROI_scale33 (used in HCP-LANGUAGE preprocessing), ROIv_scale33 (a dilated version). Other resolutions could be considered as well.
- download_flag: Download nii images from aws if set as true.
- resampling_flag: Resample niii image from 2mm to 1.25mm if set as true.
- smoothing_flag: Do spatial smoothing if set as true.  (set to false in HCP-LANGUAGE preprocessing).
- timeseries_extraction_flag: If set as true, extract BOLD time series from unsampled and possibly  smoothed nii images and then compute correlation matrices.
- overwrite_flag: If set as true, remove existing extracted time series and correlation matrices
- number of parallel threads: In line 258 of preproc_tfmri.sh, change --jobs 8 to an appropriate number (not too large to kill part of threads, for example 15).          
- parallel --jobs 8 "3dmaskdump -xyz -mask $MASK_DIR/{}.nii.gz $tfMRI_final \
             > $tfmri_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/{}.txt" ::: "${ROI_INDEX_LIST[@]}"
- Outputs: 
Each subject has a sub-directory under data directory /work/code/fs125 . Example:
--|100307
------|tfMRI
------------|tfMRI_LANGUAGE_LR.nii.gz (original image)
------------|tfMRI_LANGUAGE_RL.nii.gz
------------|tfMRI_LANGUAGE_125mm_LR.nii.gz (upsampled image)
------------|tfMRI_LANGUAGE_125mm_RL.nii.gz
------|timeseries
------------|tfMRI_LANGUAGE_125mm_LR
------------------|Lausanne2008
------------------------|ROI_scale33
------------------------------|timeseries_mean.ts (extracted mean time-series per ROI)
------------------------------|corrmat.fc (correlation matrix)
------------------------------|extracted_ts.tar.gz (archived package of meta-data)
------------|tfMRI_LANGUAGE_125mm_RL
------------------|Lausanne2008
------------------------|ROI_scale33
------------------------------|timeseries_mean.ts
------------------------------|corrmat.fc
------------------------------|extracted_ts.tar.gz
log file <path_to_data>/test_fs125_subject_list_<time_string>.txt records downloaded images for the current run.
log file <path_to_data>/downloaded_subjects_holdon.log keeps recording successfully downloaded subject images for all runs unless it gets deleted.

## Aggregate correlation matrices into a single pkl file: run ./aggregate_corrmat.py  under /work/gitrepo/py-concord-fista/preproc
Grammar: python ./aggregate_corrmat.py <path_to_data> <image_full_label> <atlas_name>
path_to_data: We use /work/code/fs125 
image_full_label: Should be consistent with subdir names under <path_to_data>/<subject_ID>/timeseries, Example: tfMRI_LANGUAGE_125mm_RL. 
atlas_name: Options could be ROI_scale33 (used in HCP-LANGUAGE preprocessing), ROIv_scale33 (a dilated version). Other resolutions could be considered as well.
Output: <path_to_data> /aggregated_corrmat_tfMRI_<task_name>_125mm_LR_<atlas_name>.p
