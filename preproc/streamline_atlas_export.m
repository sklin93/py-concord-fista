function streamline_atlas_export(atlas_file, output_file)

% example input: choose an atlas 
% atlas_file = './spark-data/final_template_1.25mm/MNI/atlas/Lausanne2008/ROI_scale33.nii.gz';

% load nifti toolbox in matlab
addpath(genpath('../nifti'));

% load atlas parcellation file
atlas = load_nii(atlas_file);

% query the ROI index of certain voxel coordinate [x,y,z]
atlas = atlas.img;
save(output_file, 'atlas', '-v7.3');

end