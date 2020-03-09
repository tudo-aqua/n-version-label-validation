# Identification of Spurious Labels in MachineLearning Data Sets using N-Version Validation

This project uses an N-version approach to validate the label quality in the BDD100k dataset.

## Requirements setup

Install the requirements in from ```requirements/requirements.txt``` using pip or conda.

## Dataset setup

#### Download the dataset at https://bdd-data.berkeley.edu/
After registering and logging in, download the dataset 'Driverable Maps' and 'Images' 
(use the 100k folder inside Images).  

- Extract the data to `data/` and arrange it to have the following directory structure:
```
data/images/train -- contains training color images
data/images/val -- contains validation color images
data/labels/train -- contains training greyscale labels 
data/labels/val -- contains validation greyscale labels 
```
- Convert the JPG images to PNG
```
./convert_to_png.sh
```
- Rename the labels, so the corresponding image and label have the same name, e.g. with this command:
```
rename s/_drivable_id// data/*/*/*.png
```
- Resize the images.
```
./resize_images.sh
```
## Train the neural networks
Train the neural networks in the following way:

```
python3 train.py --config config/FCN/bdd321x185_fcn_resnet101.yaml
python3 train.py --config config/Deeplab/bdd321x185_deeplabv3_resnet101.yaml
```
For the PSPNet, the standard setting is to only keep the last two checkpoints. Disable this by commenting out lines
247-249 in semseg/tool/train.py, so all checkpoints are kept. Then train the PSPNet in the following way.
```
cd semseg
./tool/train.sh bdd321x185 pspnet101
```  

## Evaluate model checkpoints
Run the prediction on the validation set for all model checkpoints and evaluate the results in terms of IOU. 
The final output will contain the model checkpoints with the highest mean IOU on the validation set. 
The respective predictions of the best model should be chosen for the next step. 

```
./eval_fcn_all.sh
./eval_deeplab_all.sh
```

The PSPNet training script already evaluates the mean IOU value during the training stage, so we only need to analyze
the training log to find the best checkpoint. The training log should be situated at 
`semseg/exp/bdd_321x185/pspnet101/model` and be according to the scheme `train-YYYYMMDD_HHMMSS.log`. Substitute the 
name of your own log file in the following command.
```
python3 analyze_semseg_logs.py semseg/exp/bdd_321x185/pspnet101/model/train-YYYYMMDD_HHMMSS.log semseg/exp/bdd_321x185/pspnet101/model/metrics_per_epoch.csv
```
If this should result in something that's not the last step, you will need to run the prediction for the PSPNet with 
the respective checkpoint and enter the point to which you have written the prediction images into the script mentioned 
in the next step. 

## Majority vote predictions
Enter the model checkpoints that were found to have the best IOU in the previous step at the respective 
 points in `majority_vote.sh`. 
Then run the script to execute the majority vote and sorting. This will produce images containing the 
majority vote results, number of votes for the majority class, strength of disagreement with the ground truth, 
and a visualization meant for manual inspection. Afterwards, these visualization images will be sorted into the 
different categories mentioned in the paper.  
```
./majority_vote.sh
```

# Results

Since the training process is quite time-consuming, we also provide the results of our analysis in `results`. The file 
`image_classes.csv` indicates for each image into which classes it was classified. `metrics.csv` contains the different
metrics which we used to classify images into these classes, using `majority_vote_sorting.py`.