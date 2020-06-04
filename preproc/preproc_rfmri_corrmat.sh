declare -a PHASE_ENCODING=("RL")

WORK_DIR="$1"
FLAG_SMOOTHING="$2"
FLAG_OVERWRITE="$3"
SUBJECT_LIST="$4"

ATLAS_NAME="Lausanne2008"
ATLAS_VERSION="ROIv_scale33"

fMRI_TASK="REST1"
fMRI_FILE_NAME="rfMRI_${fMRI_TASK}"

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

        # unzip timeseries
        current_dir="$PWD"
        if [ -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/extracted_ts.tar.gz ]; then 
            tar zxvf $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/extracted_ts.tar.gz --directory $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/
            echo "unzipping timeseries"
        fi

        # Calculate the averaged timeseries for each ROI
        rfMRI_ts_mean="timeseries_mean.ts"
        if [[ ! -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_ts_mean ]] || [[ $FLAG_OVERWRITE ]]; then
            if $FLAG_OVERWRITE; then 
                rm $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_ts_mean
                echo "Removing existing timeseries_mean.ts, due to FLAG_OVERWRITE turned on ..."
            fi
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
        if [[ ! -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_corrmat ]] || [[ $FLAG_OVERWRITE ]]; then
            if $FLAG_OVERWRITE; then 
                rm $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/$rfMRI_corrmat
                echo "Removing existing corrmat, due to FLAG_OVERWRITE turned on ..."
            fi
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
        if [ -f $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/extracted_ts.tar.gz ]; then 
            rm $rfMRI_ts_dir/$ATLAS_NAME/$ATLAS_VERSION/*.txt 
        fi

    done
done < $SUBJECT_LIST
