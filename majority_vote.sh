THREADS=2

PRED_DIR_1=data/bdd/bdd100k_dataset_321x185/output/predictions/deeplabv3_resnet101_preprocessed/ep70/predictions
PRED_DIR_2=data/bdd/bdd100k_dataset_321x185/output/predictions/fcn_resnet101_preprocessed/ep75/predictions
PRED_DIR_3=semseg/exp/bdd_321x185/pspnet101/result/epoch_100/val/ss/gray

TRUTH_DIR=data/bdd/bdd100k_dataset_321x185/val/labels
COLOR_IMG_DIR=data/bdd/bdd100k_dataset_321x185/val/images
OUTPUT_DIR=data/bdd/bdd100k_dataset_321x185/output/majority_vote_results

python compute_model_correlation.py ${OUTPUT_DIR}/model_correlations.csv ${PRED_DIR_1} ${PRED_DIR_2} ${PRED_DIR_3}
python majority_vote_predictions.py -t ${THREADS} ${TRUTH_DIR} ${COLOR_IMG_DIR} ${OUTPUT_DIR} ${PRED_DIR_1} ${PRED_DIR_2} ${PRED_DIR_3}
python majority_vote_sorting.py ${OUTPUT_DIR}/metrics.csv ${OUTPUT_DIR}
