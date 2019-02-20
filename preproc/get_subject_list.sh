# run the following scripts ion salinas.cs.ucsb.edu to get the subject list
SUBJECT_FILE="$1"

DSI_DIR="/local/home/share/DSI_data/fs125/mda_fibs"
DSI_LIST=$(find $DSI_DIR -mindepth 1 -maxdepth 1 -type d)

for subject_dir in $DSI_LIST
do
    subject_id=${subject_dir##*/}
    if [[ $subject_id =~ [0-9]{6}$ ]]; then
            echo $subject_id >> $SUBJECT_FILE
    fi
done



