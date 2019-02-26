# Extract time-series from tfMRI

# argument example:
# ATLAS_NAME="Lausanne2008"
# ATLAS_VERSION="ROIv_scale33"
# MASK_LIST="mask_list.txt"

WORK_DIR="$1"
ATLAS_NAME="$2"
ATLAS_VERSION="$3"
MASK_LIST="$4"

MASK_DIR="$WORK_DIR/atlas_mask/$ATLAS_NAME/$ATLAS_VERSION"

Timeseries_Dir="$TS_DIR/$ATLAS_NAME/$ATLAS_VERSION"


# get the number of ROIs in the selected atlas
Intensity_Max=`fslstats ${ATLAS_FILE} -R | cut -d " " -f 2 `
ROI_Num=${Intensity_Max%.*}
if [ -d "${MASK_DIR}" ]; then
    echo "    Atlas:${Atlas}/${Atlas_Ver}.nii.gz contains ${ROI_Num} ROI regions."
fi




# extract time courses for each ROI
mkdir -p ${WD}/atlas_ts/${Atlas}/${Atlas_Ver}
timestamp # print timestamp

echo "    ====== START: extracting time-series from tfMRI (${WD}) ======"
## parallel for loop example:
## non-parallel: 
##	for i in 1 2 3 4 5; do someCommand data$i.fastq > output$i.txt & done
## parallel:
##	parallel --jobs 16 someCommand data{}.fastq '>' output{}.fastq ::: {1..512}

# - - - - - parallel version - - - - - - 
index_list=`seq 1 $ROI_Num`
parallel --jobs 6 fslmeants -i ${Final_fMRI} -o ${Timeseries_Dir}/{}.txt -m ${MASK_DIR}/{}.nii.gz ::: "${index_list[@]}"

## - - - - - non-parallel version - - - - - - 
## for ((k=1;k<=${ROI_Num};k++))
## do
## {
## 	echo "        ROI: ${k}"
## 	# fslstats ${Final_fMRI} -k ${MASK_DIR}/${k}.nii.gz -m > ${Timeseries_Dir}/${k}_fslstats.txt
## 	fslmeants -i ${Final_fMRI} -o ${Timeseries_Dir}/${k}.txt -m ${MASK_DIR}/${k}.nii.gz
## }
## done

# - - - - - - merge single ROI ts file into a unified file - - - - - -
file_string=`cat ${MASK_DIR}/${MASK_LIST}`
cd ${Timeseries_Dir}
paste ${file_string} > ${fMRI_task}.ts.csv
tar zcvf ${fMRI_task}.tar.gz *.txt
rm *.txt
cd ${Current_Path}
## echo `pwd`

timestamp # print timestamp
echo "    ====== END: extracting time-series from tfMRI (${WD}) ======"
