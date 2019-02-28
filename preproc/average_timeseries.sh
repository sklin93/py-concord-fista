# Extract time-series from tfMRI

# argument example:
# ATLAS_NAME="Lausanne2008"
# ATLAS_VERSION="ROIv_scale33"
# MASK_LIST="mask_list.txt"

WORK_DIR="$1"
TS_DIR="$2"
ATLAS_NAME="$3"
ATLAS_VERSION="$4"
TFMRI_FINAL="$5"


# get the number of ROIs in the selected atlas
ATLAS_DIR="$WORK_DIR/final_template_1.25mm/MNI"
ATLAS_FILE_NAME="$ATLAS_DIR/atlas/$ATLAS_NAME/$ATLAS_VERSION.nii.gz"
Intensity_Max=`fslstats ${ATLAS_FILE_NAME} -R | cut -d " " -f 2 `
ROI_NUM=${Intensity_Max%.*}
MASK_DIR="$WORK_DIR/atlas_mask/$ATLAS_NAME/$ATLAS_VERSION"
if [ -d "${MASK_DIR}" ]; then
    echo "    Atlas:$ATLAS_NAME/$ATLAS_VERSION.nii.gz contains ${ROI_NUM} ROI regions."
fi


# remove the all-zero row 




# calculate the average across ROI voxels






# - - - - - - merge single ROI ts file into a unified file - - - - - -



file_string=`cat ${MASK_DIR}/${MASK_LIST}`
cd ${Timeseries_Dir}
paste ${file_string} > ${fMRI_task}.ts.csv
tar zcvf ${fMRI_task}.tar.gz *.txt
rm *.txt
cd ${Current_Path}
## echo `pwd`
