# create structural matrices based on streamline tracking

WD_In="$1"
WD_Out="$2"
Subject_ID="$3"
Atlas="$4"
Atlas_Ver="$5"
DSI_Prefix="$6"



# WD_In="../example"
# WD_Out="/work/code/data-STFN/template_subjects"
# Subject_ID="111312"
# Atlas="Lausanne2008"
# Atlas_Ver="ROI_scale33"
# DSI_Prefix="Diffusion"



timestamp() {
  date +"%T"
}


echo "    ====== START: creating structural matrix (${Subject_ID}) ======"
timestamp # print timestamp



# export atlas/parcellation matrix to text file
Template_Dir="../spark-data/final_template_1.25mm/MNI"
Atlas_File="${Template_Dir}/atlas/${Atlas}/${Atlas_Ver}.nii.gz"
Atlas_Text="${Template_Dir}/atlas/${Atlas}/${Atlas_Ver}.mat"

if [ ! -f "${Atlas_Text}" ]; then
	Current_Path=`pwd` # -- save current path
	echo "cd ${Current_Path}; streamline_atlas_export('${Atlas_File}', '${Atlas_Text}'); "
	matlab -nodesktop -nosplash -r "cd ${Current_Path}; streamline_atlas_export('${Atlas_File}', '${Atlas_Text}'); exit;"
fi



# streamline files of each subject 
# (normalized to FS125 space/template, 1.25mm)
# Streamline_Dir="/spark-data/normalized/tracking"


# create output directory
mkdir -p ${WD_Out}/${Subject_ID}
mkdir -p ${WD_Out}/${Subject_ID}/${DSI_Prefix}
mkdir -p ${WD_Out}/${Subject_ID}/${DSI_Prefix}/fs125

Streamline_File="${WD_In}/${Subject_ID}/${Subject_ID}_fs125_streams.h5"
Streamline_Out_Dir="${WD_Out}/${Subject_ID}/${DSI_Prefix}/fs125"
Streamline_Out_Prefix="${Streamline_Out_Dir}/${Subject_ID}_fs125_stmat"
Streamline_Log_File="${Streamline_Out_Dir}/${Subject_ID}.pylog"


# extract structural matrices from stramline tracking data
# echo "python ./streamline_to_matrix.py ${Streamline_File} ${Atlas_Text} ${Streamline_Out_Prefix} > ${Streamline_Log_File}"
python ./streamline_to_matrix.py ${Streamline_File} ${Atlas_Text} ${Streamline_Out_Prefix} > ${Streamline_Log_File}


# Streamline_Mat_All="${Streamline_Out_Prefix}.a.csv"
# Streamline_Mat_Qaulified="${Streamline_Out_Prefix}.q.csv"


timestamp # print timestamp
echo "    ====== END: creating structural matrix (${Subject_ID}) ======"
