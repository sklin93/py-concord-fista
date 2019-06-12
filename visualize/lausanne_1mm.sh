#!/bin/bash
# run this script under /visualize directory

# export SUBJECTS_DIR=visualize/Yeo_JNeurophysiol11_MNI152
easy_lausanne \
    --subject_id HCP1.25_template \
    --target_volume ./Yeo_JNeurophysiol11_MNI152/FSL_MNI152_FreeSurferConformed_1mm.nii.gz \
    --target_type anisotropy \
    --output_dir ./Lausanne2008/1mm \
    --include500




Lausanne_atlas_1.25mm=
Lausanne_atlas_1mm=
FSL_MNItemplate_1mm=Yeo_JNeurophysiol11_MNI152/FSL_MNI152_FreeSurferConformed_1mm.nii.gz

flirt -in brain_1.25mm.nii.gz -ref $FSL_MNItemplate_1mm -out brain_1mm.nii.gz -applyisoxfm