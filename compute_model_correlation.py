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
import glob
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from utils import path_join

parser = argparse.ArgumentParser(
    description='This script takes several directories, each containing the '
                'predictions made by a different model, and '
                'computes a confusion matrix, indicating how often '
                'which models agree with each other in their predictions. '
                'If a majority vote sorting is provided, the confusion matrix '
                'can instead be computed per category.')
parser.add_argument('log_file',
                    help='CSV-file to which the confusion matrix is written')
parser.add_argument('prediction_dirs', nargs='*',
                    help='Variable number of directories containing png files '
                         'representing segmentation predictions for '
                         'the same pictures, named exactly the same as '
                         'the respective ground truth labels')
parser.add_argument('-m', '--mv_sorting',
                    help='Path to a majority vote sorting file output by '
                         'majority_vote_sorting.py'
                         'If set, the model correlation is computed per '
                         'category in the sorting file.')
args = parser.parse_args()

n_preds = len(args.prediction_dirs)
if n_preds < 2:
    print('Please specify at least two directories '
          'containing predictions to be compared.')
    exit(1)

if args.mv_sorting is None:

    pred_paths = [sorted(glob.glob(path_join(args.prediction_dirs[i], '*.png')))
                  for i in range(n_preds)]
    equal_lengths = [len(pred_paths[i]) == len(pred_paths[0]) for i in
                     range(1, n_preds)]
    if not all(equal_lengths):
        print('Different number of images in the given prediction directories.')
        exit(1)

    conf_mat = np.ones((n_preds, n_preds))
    for model_i in range(n_preds):
        for model_j in range(model_i + 1, n_preds):
            acc_list = []
            for img_path_i, img_path_j in zip(pred_paths[model_i],
                                              pred_paths[model_j]):
                print(f'Comparing {Path(img_path_i).name} '
                      f'for models {model_i} and {model_j}...')
                img_i = cv2.imread(img_path_i, cv2.IMREAD_GRAYSCALE)
                img_j = cv2.imread(img_path_j, cv2.IMREAD_GRAYSCALE)

                acc = np.mean((img_i == img_j))
                acc_list.append(acc)

            conf_mat[model_i, model_j] = np.mean(acc_list)
            conf_mat[model_j, model_i] = np.mean(acc_list)

    conf_mat_df = pd.DataFrame(conf_mat, index=args.prediction_dirs,
                               columns=args.prediction_dirs)
    conf_mat_df.to_csv(args.log_file)

else:
    mv_sorting_df = pd.read_csv(args.mv_sorting)

    # Init confusion matrices
    categories = list(
        mv_sorting_df.drop(['Ground truth', 'Visualization', 'Color image'],
                           axis=1).columns)
    category_to_conf_mat = dict()
    for category in categories:
        conf_mat = np.ones((n_preds, n_preds))
        category_df = mv_sorting_df[mv_sorting_df[category] == 1][
            'Ground truth']
        print(category, category_df.shape)
        # For each pair of models, compare their predictions on all images
        # and compute the mean rate of agreement
        for model_i in range(n_preds):
            for model_j in range(model_i + 1, n_preds):
                acc_list = []
                for _, img_path in category_df.iteritems():
                    img_name = Path(img_path).name
                    img_i = cv2.imread(
                        f'{args.prediction_dirs[model_i]}/{img_name}',
                        cv2.IMREAD_GRAYSCALE)
                    img_j = cv2.imread(
                        f'{args.prediction_dirs[model_j]}/{img_name}',
                        cv2.IMREAD_GRAYSCALE)

                    acc = np.mean((img_i == img_j))
                    acc_list.append(acc)

                conf_mat[model_i, model_j] = np.mean(acc_list)
                conf_mat[model_j, model_i] = np.mean(acc_list)

        category_to_conf_mat[category] = conf_mat

    with open(args.log_file, 'w') as f:
        f.write(f'Order of prediction directories used '
                f'in the confusion matrices: {args.prediction_dirs}\n')
        for category in categories:
            f.write(f'\n\nConfusion matrix for category {category}: \n')
            f.write(str(category_to_conf_mat[category]))
            print('\n\n', category, '\n')
            print(category_to_conf_mat[category])
