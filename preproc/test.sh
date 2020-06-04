# ./test.sh --workpath=/work/code/fs125_rs --task=REST1 --atlas=ROIv_scale33  --download=true --upsampling=true --smoothing=false --bpfiltering=true --tsextract=true --overwrite=false

for i in "$@"
do
case $i in
    -w=*|--workpath=*)
    WORK_DIR="${i#*=}"
    shift # past argument=value
    ;;
    -t=*|--task=*)
    fMRI_TASK="${i#*=}"
    shift # past argument=value
    ;;
    -a=*|--atlas=*)
    ATLAS_VERSION="${i#*=}"
    shift # past argument=value
    ;;
    -d=*|--download=*)
    FLAG_DOWNLOAD="${i#*=}"
    shift # past argument=value
    ;;
    -u=*|--upsampling=*)
    FLAG_UPSAMPING="${i#*=}"
    shift # past argument=value
    ;;
    -s=*|--smoothing=*)
    FLAG_SMOOTHING="${i#*=}"
    shift # past argument=value
    ;;
    -f=*|--bpfiltering=*)
    FLAG_BPFILTERING="${i#*=}"
    shift # past argument=value
    ;;
    -e=*|--tsextract=*)
    FLAG_TSEXTRACT="${i#*=}"
    shift # past argument=value
    ;;
    -o=*|--overwrite=*)
    FLAG_OVERWRITE="${i#*=}"
    shift # past argument=value
    ;;
    --default)
    DEFAULT=YES
    shift # past argument with no value
    ;;
    *)
          # unknown option
    ;;
esac
done

echo "Input arguments:"
echo "WORK_DIR       = ${WORK_DIR}"
echo "fMRI_TASK      = ${fMRI_TASK}"
echo "ATLAS_VERSION  = ${ATLAS_VERSION}"

echo "FLAG_DOWNLOAD    = ${FLAG_DOWNLOAD}"
echo "FLAG_UPSAMPING   = ${FLAG_UPSAMPING}"
echo "FLAG_SMOOTHING   = ${FLAG_SMOOTHING}"
echo "FLAG_BPFILTERING = ${FLAG_BPFILTERING}"
echo "FLAG_TSEXTRACT   = ${FLAG_TSEXTRACT}"
echo "FLAG_OVERWRITE   = ${FLAG_OVERWRITE}"