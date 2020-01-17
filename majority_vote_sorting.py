#!/usr/bin/env python3
#
# Copyright, TU Dortmund 2020
#
# MIT License
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
import argparse
from shutil import copyfile

import matplotlib.pyplot as plt
import pandas as pd

from utils import *


def copy_to(original_path, target_dir, do_copy=True):
    if do_copy:
        copyfile(original_path,
                 f'{args.output_dir}/{target_dir}/' + Path(original_path).name)


parser = argparse.ArgumentParser(
    description='This script sorts images into different '
                'error class directories, based on the given '
                'majority vote metrics.')
parser.add_argument('metrics_file',
                    help='File output by majority_vote_predictions.py, '
                         'containing metrics and file paths.')
parser.add_argument('output_dir',
                    help='Directory in which the error class directories '
                         'are created and to which images are copied.')
parser.add_argument('--copy_images',
                    action='store_true',
                    help='If set, the visualization images will be copied to '
                         'the output directory. Otherwise, only a CSV file '
                         'will be created which contains the classification.')
args = parser.parse_args()

metrics_df = pd.read_csv(args.metrics_file)

Path(args.output_dir + '/' + DIFFICULT_SITUATIONS).mkdir(
    parents=True, exist_ok=True)
Path(args.output_dir + '/' + LANES_SWAPPED).mkdir(
    parents=True, exist_ok=True)
Path(args.output_dir + '/' + MV_BG_LABEL_DISAGREE).mkdir(
    parents=True, exist_ok=True)
Path(args.output_dir + '/' + LABEL_BG_MV_AGREE).mkdir(
    parents=True, exist_ok=True)
Path(args.output_dir + '/' + LABEL_BG_MV_DISAGREE).mkdir(
    parents=True, exist_ok=True)
Path(args.output_dir + '/' + STRONG_AGREE).mkdir(parents=True, exist_ok=True)
Path(args.output_dir + '/' + CROSSING).mkdir(parents=True, exist_ok=True)

classifications = []
for _, row in metrics_df.iterrows():
    classification = pd.Series(row[['Ground truth', 'Visualization',
                                    'Color image']])

    # Find images where prediction is almost completely background,
    # but label contains non-background
    if row['Ratio of background pixels in majority vote'] >= 0.999 and \
            row['Ratio of disagreeing pixels'] > 0:
        copy_to(row['Visualization'], MV_BG_LABEL_DISAGREE, args.copy_images)
        classification[MV_BG_LABEL_DISAGREE] = 1
    else:
        classification[MV_BG_LABEL_DISAGREE] = 0

    # Find images where ground truth is completely background,
    # but prediction contains non-background
    if row['Ratio of background pixels in ground truth'] == 1.0 and \
            row['Ratio of disagreeing pixels'] > 0:
        copy_to(row['Visualization'], LABEL_BG_MV_DISAGREE, args.copy_images)
        classification[LABEL_BG_MV_DISAGREE] = 1
    else:
        classification[LABEL_BG_MV_DISAGREE] = 0

    # Find images where ground truth is completely background
    # and prediction agrees
    if row['Ratio of background pixels in ground truth'] == 1.0 and \
            row['Ratio of disagreeing pixels'] == 0:
        copy_to(row['Visualization'], LABEL_BG_MV_AGREE,
                args.copy_images)
        classification[LABEL_BG_MV_AGREE] = 1
    else:
        classification[LABEL_BG_MV_AGREE] = 0

    if row['Ratio of disagreeing pixels'] <= 0.065:
        copy_to(row['Visualization'], STRONG_AGREE, args.copy_images)
        classification[STRONG_AGREE] = 1
    else:
        classification[STRONG_AGREE] = 0

    # Find images where the different predictions disagree a lot
    if row['Mean divergent votes'] >= 0.1:
        copy_to(row['Visualization'], DIFFICULT_SITUATIONS, args.copy_images)
        classification[DIFFICULT_SITUATIONS] = 1
    else:
        classification[DIFFICULT_SITUATIONS] = 0

    # Find images where the prediction is improved
    # by swapping the two lane classes
    if row['Ratio of disagreeing pixels'] - row['Swapped lanes disagreement'] \
            >= 0.01:
        copy_to(row['Visualization'], LANES_SWAPPED, args.copy_images)
        classification[LANES_SWAPPED] = 1
    else:
        classification[LANES_SWAPPED] = 0

    # Find crossing
    if row['Height of horizontal rectangle'] >= 0.1:
        copy_to(row['Visualization'], CROSSING, args.copy_images)
        classification[CROSSING] = 1
    else:
        classification[CROSSING] = 0

    classifications.append(classification)

df = pd.DataFrame(classifications)
df.to_csv(args.output_dir + '/image_classes.csv', index=None)

# Visualize distribution of categories via bar plots
df = df.rename(
    {MV_BG_LABEL_DISAGREE: 'MV: all BG, GT disagrees',
     LABEL_BG_MV_DISAGREE: 'GT: All BG, MV disagrees',
     LABEL_BG_MV_AGREE: 'GT: All BG, MV agrees',
     STRONG_AGREE: 'Strong agreement',
     DIFFICULT_SITUATIONS: 'Difficult situations',
     LANES_SWAPPED: 'Lanes swapped', CROSSING: 'Crossing'}, axis=1)
plot = df.drop(['Ground truth', 'Visualization', 'Color image'], axis=1).sum(
    axis=0).plot.bar()
plt.xticks(rotation=20)
plt.tight_layout()
figure = plot.get_figure()
figure.savefig(args.output_dir + '/all_classes_plot.png')

plt.clf()
plot = df.drop(
    ['Ground truth', 'Visualization', 'Color image', 'Strong agreement'],
    axis=1).sum(axis=0).plot.bar()
plt.xticks(rotation=20)
plt.tight_layout()
figure = plot.get_figure()
figure.savefig(args.output_dir + '/classes_plot_without_strong_agreement.png')
