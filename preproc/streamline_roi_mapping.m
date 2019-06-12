function streamline_roi_mapping(atlas_file)

atlas_file = './spark-data/final_template_1.25mm/MNI/atlas/Lausanne2008/ROI_scale33.nii.gz';

% load nifti toolbox in matlab
addpath(genpath('../nifti'));

% load atlas parcellation file
atlas = load_nii('atlas_file');

% query the ROI index of certain voxel coordinate [x,y,z]
atlas.img(x,y,z);

end