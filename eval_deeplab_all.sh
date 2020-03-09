MODEL_DIR=data/bdd/bdd100k_dataset_321x185/output/predictions/deeplabv3_resnet101_preprocessed
VAL_DIR=data/bdd/bdd100k_dataset_321x185/val
CONFIG_FILE=config/Deeplab/bdd321x185_deeplabv3_resnet101.yaml
ARCH=deeplab
THREADS=4
N_CLASSES=3

./eval.sh $MODEL_DIR $VAL_DIR $CONFIG_FILE $ARCH $THREADS $N_CLASSES
