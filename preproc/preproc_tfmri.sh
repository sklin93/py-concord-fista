### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ####
# Preparing

# working directory where you save your metadata and final results
WORK_DIR="$1"

# phase encoding options
declare -a PHASE_ENCODING=("LR" "RL")



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

fMRI_TASK="LANGUAGE"
fMRI_FILE_NAME="tfMRI_${fMRI_TASK}"
SUBJECT_LIST=$WORK_DIR/$SUBJECT_FILE_NAME

while read -r subject;
do
    echo "Step 1: Subject $subject ......"
    mkdir -p $WORK_DIR/$subject/tfMRI
    for phase in "${PHASE_ENCODING[@]}"
    do
        file_relative_path=MNINonLinear/Results/${fMRI_FILE_NAME}_$phase/${fMRI_FILE_NAME}_$phase.nii.gz
        if [ ! -f $WORK_DIR/$subject/tfMRI/${fMRI_FILE_NAME}_$phase.nii.gz ]; then
            aws s3 cp s3://hcp-openaccess/HCP/$subject/${file_relative_path} \
                $WORK_DIR/$subject/tfMRI --region us-west-2
        else
            echo "Existing tfMRI file found: ${fMRI_FILE_NAME}_$phase.nii.gz"
        fi
    done
done < $SUBJECT_LIST



### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# AFNI 3dresample, resample 2mm fMRI to 1.25mm

while read -r subject;
do
    echo "Step 2: Subject $subject ......"
    for phase in "${PHASE_ENCODING[@]}"
    do
        tfMRI_2mm=$WORK_DIR/$subject/tfMRI/${fMRI_FILE_NAME}_$phase.nii.gz
        tfMRI_125mm=$WORK_DIR/$subject/tfMRI/${fMRI_FILE_NAME}_125mm_$phase.nii.gz
        if [ ! -f $tfMRI_125mm ]; then
            echo "Downsampling started, input: $tfMRI_2mm"
            3dresample -dxyz 1.25 1.25 1.25 -orient LPI \
                -inset $tfMRI_2mm -prefix $tfMRI_125mm -overwrite
            echo "Downsampling finished, output: $tfMRI_125mm"
        else
            echo "Existing downsampled tfMRI file found: $tfMRI_125mm"
        fi
    done
done < $SUBJECT_LIST

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






### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# Create masks from atlas definition nii image

ATLAS_DIR="Lausanne2008"
Atlas_VER="ROIv_scale33"









### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ### ###
# extract timeseries according to given atlas

# REFERENCE_TEMPLATE="final_template_1.25mm/MNI/RAS_MNI_1.25mm.nii.gz"
