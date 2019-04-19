# Usage exapmles:
# ./preproc_rfMRI.sh ~/localrepo/glasso/fs125 LANGUAGE ROIv_scale33 false false false true
# ./preproc_rfMRI.sh /work/code/fs125 LANGUAGE ROIv_scale33 false false false true

# todo tasks: 
# 1. parallel downloading and unsampling steps
# 2. enable the resumable pipeline, maintain the downloaded subject list 
# whenever there is no need to downlaod another time for preprocessing 
# with a parcellation in a different scale or resolution.


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####
# Preparing

# working directory where you save your metadata and final results
WORK_DIR="$1"
fMRI_TASK="$2" # fMRI_TASK="LANGUAGE"
ATLAS_VERSION="$3" # ATLAS_VERSION="ROIv_scale33"

# phase encoding options
declare -a PHASE_ENCODING=("LR" "RL")

# make shell scripts runnable under current directory
chmod +x *.sh

# enable FLAGS
FLAG_DOWNLOAD="$4"
FLAG_UPSAMPING="$5"
FLAG_SMOOTHING="$6"
FLAG_TSEXTRACT="$7"

time_start_all_steps="$(date -u +%s)"

### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# get 630 DSI subject list (IDs) from salinas.cs.ucsb.edu

DSI_SERVER_NAME="salinas.cs.ucsb.edu"
SUBJECT_FILE_NAME="test_fs125_subject_list.txt"

# check if the list already exits
if [ ! -f $WORK_DIR/$SUBJECT_FILE_NAME ]; then
	scp ./get_subject_list.sh $DSI_SERVER_NAME:~
	ssh $DSI_SERVER_NAME "chmod +x *.sh"
	ssh $DSI_SERVER_NAME "./get_subject_list.sh $SUBJECT_FILE"
	scp $DSI_SERVER_NAME:~/$SUBJECT_FILE_NAME $WORK_DIR/
else
	echo "Existing subject list found: $WORK_DIR/$SUBJECT_FILE_NAME"
fi



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# For each subject, download its task fmri file
# Need to install awscli and configure with access key pairs

fMRI_FILE_NAME="rfMRI_${fMRI_TASK}"
FULL_SUBJECT_LIST=$WORK_DIR/$SUBJECT_FILE_NAME
LOG_LIST=$WORK_DIR/"processed_subject_list.log"
SUBJECT_LIST=$WORK_DIR/"downloaded_subject_list.txt"

if $FLAG_DOWNLOAD; then
    if [ -f $SUBJECT_LIST ]; then rm $SUBJECT_LIST; fi 
    while read -r subject;
    do
        echo "Step 1: Subject $subject ......"
        mkdir -p $WORK_DIR/$subject/rfMRI
        download_success="true"
        for phase in "${PHASE_ENCODING[@]}"
        do
            file_relative_path=MNINonLinear/Results/${fMRI_FILE_NAME}_$phase/${fMRI_FILE_NAME}_$phase.nii.gz
            if [ ! -f $WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_$phase.nii.gz ]; then
                aws s3 cp s3://hcp-openaccess/HCP/$subject/${file_relative_path} \
                    $WORK_DIR/$subject/rfMRI --region us-west-2
            else
                echo "Existing rfMRI file found: ${fMRI_FILE_NAME}_$phase.nii.gz"
            fi
            # check if images are correctly downlaoded
            if [ ! -f $WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_$phase.nii.gz ]; then
                download_success="false"; 
            fi
        done
        # build a list for successfully downloaded images. Some image files do not exist on AWS.
        echo "$(date +%F)" >> $LOG_LIST
        if $download_success; then 
            echo $subject >> $SUBJECT_LIST
            echo "$subject $fMRI_TASK $ATLAS_VERSION">> $LOG_LIST
        fi
        echo $'\n'  >> $LOG_LIST
    done < $FULL_SUBJECT_LIST
else
    echo "Downloading is disabled. Check settings in script."
fi


### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# AFNI 3dresample, resample 2mm fMRI to 1.25mm

if $FLAG_UPSAMPING; then
    while read -r subject;
    do
        echo "Step 2: Subject $subject ......"
        for phase in "${PHASE_ENCODING[@]}"
        do
            rfMRI_2mm=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_$phase.nii.gz
            rfMRI_125mm=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_125mm_$phase.nii.gz
            if [ ! -f $rfMRI_125mm ]; then
                echo "Upsampling started, input: $rfMRI_2mm"
                time_start="$(date -u +%s)"
                3dresample -dxyz 1.25 1.25 1.25 -orient LPI \
                    -inset $rfMRI_2mm -prefix $rfMRI_125mm -overwrite
                time_end="$(date -u +%s)"
                time_elapsed="$(bc <<<"$time_end-$time_start")"
                echo "Upsampling finished in ${time_elapsed} seconds, output: $rfMRI_125mm"
            else
                echo "Existing upsampled rfMRI file found: $rfMRI_125mm"
            fi
        done
    done < $SUBJECT_LIST
else
    echo "Upsampling is disabled. Check settings in script."
fi

# Original NIfTI image:
# dimension:        [4 91 109 91 316 1 1 1]
# pixeldimension:   [1 2 2 2 0.7200 0 0 0]
# qoffset_x: 90
# qoffset_y: -126
# qoffset_z: -72
# srow_x: [-2 0 0 90]
# srow_y: [0 2 0 -126]
# srow_z: [0 0 2 -72]
# intent_name: ''
# magic: 'n+1'
# originator: [46 64 37 0 -32768]

# Resulting NIfTI image:
# dimension:        [4 146 174 146 316 1 1 1]
# pixeldimension:   [1 1.2500 1.2500 1.2500 0.7200 0 0 0]
# qoffset_x: -90.6250
# qoffset_y: -126.1250
# qoffset_z: -72.6250
# srow_x: [1.2500 0 0 -90.6250]
# srow_y: [0 1.2500 0 -126.1250]
# srow_z: [0 0 1.2500 -72.6250]
# intent_name: ''
# magic: 'n+1'
# originator: [73.5000 101.9000 59.1000 0 0]



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Smoothing (optional)

# Check the following paper that suggests a 2~3-voxel FWHM (6mm) spatial smoothing:
# "Effect of Spatial Smoothing on Task fMRI ICA and Functional Connectivity"

# FSL spatial smoothing command:
# fslmaths data -kernel gauss sigma -fmean smoothed.nii
# sigma = FWHM / 2.3548, and we use FWHM = 6mm or 4mm
# Check conversion between sigma and FWHM:
# https://brainder.org/2011/08/20/gaussian-kernels-convert-fwhm-to-sigma/

 # Another option is to use AFNI 3dBlurToFWHM for mpatial smoothing: 
 # http://andysbrainblog.blogspot.com/2012/06/smoothing-in-afni.html
 # 3dBlurToFWHM -FWHM 6 -automask -prefix outputDataset -input inputDataset

sigma=2.548

if $FLAG_SMOOTHING; then
    while read -r subject;
    do
        echo "Step 3 (smoothing, optional): Subject $subject ......"
        for phase in "${PHASE_ENCODING[@]}"
        do
            rfMRI_125mm=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_125mm_$phase.nii.gz
            rfMRI_125mm_smoothed=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_125mm_smoothed_$phase.nii.gz
            if [ ! -f $rfMRI_125mm_smoothed ]; then
                echo "Spatial smoothing started, input: $rfMRI_125mm"
                time_start="$(date -u +%s)"
                fslmaths $rfMRI_125mm -kernel gauss 2.548 -fmean $rfMRI_125mm_smoothed
                time_end="$(date -u +%s)"
                time_elapsed="$(bc <<<"$time_end-$time_start")"
                echo "Spatial smoothing finished in ${time_elapsed} seconds, \
                    output: $rfMRI_125mm_smoothed"
            else
                echo "Existing spatial smoothed rfMRI found: $rfMRI_125mm_smoothed"
            fi
        done
    done < $SUBJECT_LIST
else
    echo "Smoothing is disabled. Check settings in script."
fi



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create binary masks from atlas definition nii image

ATLAS_NAME="Lausanne2008"
MASK_LIST="mask_list.txt"
MASK_DIR="$WORK_DIR/atlas_mask/$ATLAS_NAME/$ATLAS_VERSION"

# check if the mask already exits
if [ ! -f $MASK_DIR/$MASK_LIST ]; then
    echo "Step 4 (creating ROI masks) ......"
    ./create_roi_mask.sh $WORK_DIR $ATLAS_NAME $ATLAS_VERSION $MASK_LIST
else
    echo "Existing masks found: $MASK_DIR/$MASK_LIST"
fi

# get the number of ROIs in the selected atlas
ATLAS_DIR="$WORK_DIR/final_template_1.25mm/MNI"
ATLAS_FILE_NAME="$ATLAS_DIR/atlas/$ATLAS_NAME/$ATLAS_VERSION.nii.gz"
Intensity_Max=`fslstats ${ATLAS_FILE_NAME} -R | cut -d " " -f 2 `
ROI_NUM=${Intensity_Max%.*}
ROI_INDEX_LIST=`seq 1 $ROI_NUM`
if [ -d "${MASK_DIR}" ]; then
    echo "    Atlas:$ATLAS_NAME/$ATLAS_VERSION.nii.gz contains ${ROI_NUM} ROI regions."
fi



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Extract timeseries given atlas masks and compute correlation matrix
# REFERENCE_TEMPLATE="final_template_1.25mm/MNI/RAS_MNI_1.25mm.nii.gz"

# Extracting Timecourses with 3dmaskdump, check:
# https://www.andysbrainblog.com/andysbrainblog/2017/5/5/extracting-timecourses-with-3dmaskdump
# Another option by using FSL command:
# parallel --jobs 6 fslmeants -i ${Final_fMRI} -o ${TS_DIR}/{}.txt -m ${MASK_DIR}/{}.nii.gz ::: "${ROI_INDEX_LIST[@]}"

if $FLAG_TSEXTRACT; then
    while read -r subject;
    do
        echo "Step 5 (extracting timeseries): Subject $subject ......"
        for phase in "${PHASE_ENCODING[@]}"
        do
            # Set up input rfMRI image dir and output dir
            if $FLAG_SMOOTHING; then
                rfMRI_final=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_125mm_smoothed_$phase.nii.gz
                rfMRI_ts_dir=$WORK_DIR/$subject/timeseries/${fMRI_FILE_NAME}_125mm_smoothed_${phase}
            else
                rfMRI_final=$WORK_DIR/$subject/rfMRI/${fMRI_FILE_NAME}_125mm_$phase.nii.gz
                rfMRI_ts_dir=$WORK_DIR/$subject/timeseries/${fMRI_FILE_NAME}_125mm_${phase}
            fi
            # Extract timeseries from rfMRI image for each ROI
            if [ ! -d $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION ]; then
                echo "Extraction started, input: $rfMRI_final"
                mkdir -p $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION
                time_start="$(date -u +%s)"
                parallel --jobs 15 "3dmaskdump -xyz -mask $MASK_DIR/{}.nii.gz $rfMRI_final \
                    > $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/{}.txt" ::: "${ROI_INDEX_LIST[@]}"
                time_end="$(date -u +%s)"
                time_elapsed="$(bc <<<"$time_end-$time_start")"
                echo "Extraction finished in ${time_elapsed} seconds, output: \
                    $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION"
            else
                echo "Existing extracted timeseries found: $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION"
            fi
            # Calculate the averaged timeseries for each ROI
            rfMRI_ts_mean="timeseries_mean.ts"
            if [ ! -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_ts_mean ]; then
                echo "Averaging timeseries started."
                python ./average_timeseries.py $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION $rfMRI_ts_mean
                echo "Averaging timeseries finished, output: \
                    $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_ts_mean"
            else
                echo "Existing averaged timeseries found: \
                    $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_ts_mean"
            fi
            # Compute the correlation
            rfMRI_corrmat="corrmat.fc"
            if [ ! -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_corrmat ]; then
                echo "Connecitivity matrix construction started."
                python ./create_corrmat.py $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION \
                    $rfMRI_ts_mean $rfMRI_corrmat
                echo "Connecitivity matrix construction finished, output: \
                    $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_corrmat"
            else
                echo "Existing correlation matrix found: \
                    $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_corrmat"
            fi
            # Archive timeseries
            current_dir="$PWD"
            cd $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/
            if [ -f extracted_ts.tar.gz ]; then rm extracted_ts.tar.gz; fi
            tar zcvf extracted_ts.tar.gz *.txt && rm ./*.txt
            cd $current_dir
        done
    done < $SUBJECT_LIST
else
    echo "Timeseries extraction is disabled. Check settings in script."
fi


time_end_all_steps="$(date -u +%s)"
time_elapsed_all_steps="$(bc <<<"$time_end_all_steps-$time_start_all_steps")"
echo "All steps finished in ${time_elapsed_all_steps} seconds."