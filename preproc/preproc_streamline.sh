#!/bin/bash

DSI_Prefix="Diffusion"
Atlas="Lausanne2008"
Atlas_Ver="ROI_scale33"


dir_name="/store/fs125/normalized/tracking"
dir_name_out="/work/code/data-STFN/template_subjects"
dir_list=$(find $dir_name -mindepth 1 -maxdepth 1 -type d)


j=0
nb_concurrent_processes=14
for subject_dir in $dir_list
do
	subject_id=${subject_dir##*/}
	echo $subject_id

	mat_name="$dir_name_out/$subject_id/$DSI_Prefix/$Atlas/$Atlas_Ver/${subject_id}_fs125_stmat.a.csv"
	# if [ ! -f "${mat_name}" ]; then
	echo "Processing: $subject_id = = = = = = = = = ="
	./streamline_to_matrix.sh $dir_name $dir_name_out $subject_id $Atlas $Atlas_Ver $DSI_Prefix &
	((++j == nb_concurrent_processes)) && { j=0; wait; }
	# else
	# 	echo "Processed: $subject_id = = = = = = = = = ="
	# fi
done