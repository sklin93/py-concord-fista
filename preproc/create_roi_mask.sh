#!/bin/bash

# Creating masks from the given atlas definition nii image
# Check https://www.andysbrainblog.com/andysbrainblog/2017/6/2/creating-masks-from-the-juelich-atlas for creating binary masks using fslmaths
# Check http://andysbrainblog.blogspot.com/2012/11/creating-masks-in-fsl.html for manually creating masks in FSL

# argument example:
# ATLAS_NAME="Lausanne2008"
# ATLAS_VERSION="ROI_scale33"
# MASK_LIST="file_string.txt"

WORK_DIR="$1"
ATLAS_NAME="$2"
ATLAS_VERSION="$3"
MASK_LIST="$4"

ATLAS_DIR="$WORK_DIR/final_template_1.25mm/MNI"
ATLAS_FILE_NAME="$ATLAS_DIR/atlas/$ATLAS_NAME/$ATLAS_VERSION.nii.gz"


MASK_DIR="$WORK_DIR/atlas_mask/$ATLAS_NAME/$ATLAS_VERSION"
mkdir -p $MASK_DIR

# get the number of ROIs in the selected atlas
Intensity_Max=`fslstats $ATLAS_FILE_NAME.nii.gz -R | cut -d " " -f 2 `
roi_num=${Intensity_Max%.*}
echo "- - - Atlas:$ATLAS_NAME/$ATLAS_VERSION.nii.gz contains $roi_num ROI regions."

# generate binary masks for each ROI in the atlas
# fslmaths -thr: lower threshold, -uthr: upper threshold, -bin: binary mask
k=1
while [ $k -le ${roi_num} ]
do
	fslmaths $ATLAS_FILE_NAME -thr $k -uthr $k -bin $MASK_DIR/${k}
	let k=k+1 
done
echo "- - - Atlas: generating masks for each ROI in the atlas"

# create meta file string
file_string=""
for ((k=1;k<=$roi_num;k++))
do
	file_string="${file_string} ${k}.txt"
done
echo ${file_string} > $MASK_DIR/$MASK_LIST
echo "- - - Atlas: creating meta file string"