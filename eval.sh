MODEL_DIR=$1
VAL_DIR=$2
CONFIG_FILE=$3
ARCH=$4
THREADS=$5

echo "Starting evaluation run with config"
echo "MODEL_DIR $MODEL_DIR"
echo "VAL_DIR $VAL_DIR"
echo "CONFIG_FILE $CONFIG_FILE"
echo "ARCH $ARCH"
echo "THREADS $THREADS"
echo "N_CLASSES $N_CLASSES"

for i in {0..95..5}
do
    mkdir ${MODEL_DIR}/ep${i}/
    python predict.py ${MODEL_DIR}/model_ep${i}.dict ${VAL_DIR} ${MODEL_DIR}/ep${i}/ ${CONFIG_FILE} -a ${ARCH} --n_classes ${N_CLASSES}
    python compute_iou_parallel.py ${MODEL_DIR}/ep${i}/predictions ${VAL_DIR}/labels ${MODEL_DIR}/ep${i}/iou.csv -t ${THREADS} --n_classes ${N_CLASSES}
done

mkdir ${MODEL_DIR}/ep99/
python predict.py ${MODEL_DIR}/model_ep99.dict ${VAL_DIR} ${MODEL_DIR}/ep99/ ${CONFIG_FILE} -a ${ARCH} --n_classes ${N_CLASSES}
python compute_iou_parallel.py ${MODEL_DIR}/ep99/predictions ${VAL_DIR}/labels ${MODEL_DIR}/ep99/iou.csv -t ${THREADS} --n_classes ${N_CLASSES}

python analyze_torchvision_logs.py ${MODEL_DIR} --output_file ${MODEL_DIR}/iou_per_ep.csv
