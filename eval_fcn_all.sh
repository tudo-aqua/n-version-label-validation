MODEL_DIR=/data/bdd/bdd100k_dataset_321x185/output/predictions/fcn_resnet101_preprocessed
VAL_DIR=/data/bdd/bdd100k_dataset_321x185/val
CONFIG_FILE=config/FCN/bdd321x185_fcn_resnet101.yaml
ARCH=fcn
THREADS=10
N_CLASSES=3

./eval.sh $MODEL_DIR $VAL_DIR $CONFIG_FILE $ARCH $THREADS $N_CLASSES
