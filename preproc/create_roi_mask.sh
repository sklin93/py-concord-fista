#!/bin/bash

# argument example
# Atlas="Lausanne2008"
# Atlas_Ver="ROI_scale33"
# File_String_Name="file_string.txt"

Atlas="$1"
Atlas_Ver="$2"
File_String_Name="$3"

Template_Dir="../spark-data/final_template_1.25mm/MNI"
Atlas_File="${Template_Dir}/atlas/${Atlas}/${Atlas_Ver}.nii.gz"
Atlas_MaskDir="${Template_Dir}/atlas_mask/${Atlas}/${Atlas_Ver}"

mkdir -p ${Template_Dir}/atlas_mask/${Atlas}/${Atlas_Ver}

# get the number of ROIs in the selected atlas
Intensity_Max=`fslstats ${Atlas_File}.nii.gz -R | cut -d " " -f 2 `
ROI_Num=${Intensity_Max%.*}
echo "    Atlas:${Atlas}/${Atlas_Ver}.nii.gz contains ${ROI_Num} ROI regions."

# generate binary masks for each ROI in the atlas
# fslmaths -thr: lower threshold, -uthr: upper threshold, -bin: binary mask
k=1
while [ $k -le ${ROI_Num} ]
do
	fslmaths ${Atlas_File} -thr $k -uthr $k -bin ${Atlas_MaskDir}/${k}
	let k=k+1 
done
echo "    Atlas: generating masks for each ROI in the atlas"

# create meta file string
file_string=""
for ((k=1;k<=${ROI_Num};k++))
do
	file_string="${file_string} ${k}.txt"
done
echo ${file_string} > ${Atlas_MaskDir}/${File_String_Name}
echo "    Atlas: creating meta file string"