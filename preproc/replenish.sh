declare -a PHASE_ENCODING=("LR" "RL")

SUBJECT_LIST="/work/code/fs125/replenish_subject_list.txt"

while read -r subject;
do
    for phase in "${PHASE_ENCODING[@]}"
    do
        output_dir="/work/code/fs125/$subject/timeseries/tfMRI_LANGUAGE_125mm_$phase/Lausanne2008/ROI_scale33"
        3dmaskdump -xyz -mask $output_dir/8.nii.gz /work/code/fs125/$subject/tfMRI/tfMRI_LANGUAGE_125mm_$phase.nii.gz > $output_dir/8.txt
        [ -f $output_dir/timeseries_mean.ts ] && rm $output_dir/timeseries_mean.ts
        [ -f $output_dir/corrmat.fc ] && rm $output_dir/corrmat.fc
        python ./average_timeseries.py $output_dir timeseries_mean.ts
        python ./create_corrmat.py $output_dir timeseries_mean.ts corrmat.fc
        current_dir="$PWD"
        cd $output_dir
        if [ -f extracted_ts.tar.gz ]; then rm extracted_ts.tar.gz; fi
        tar zcvf extracted_ts.tar.gz *.txt && rm ./*.txt
        cd $current_dir
    done
done < $SUBJECT_LIST
