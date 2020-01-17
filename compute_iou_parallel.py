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
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from per_class_iou import per_class_iou
from utils import path_join

parser = argparse.ArgumentParser(
    description='This script evaluates the per-class iou '
                'for given predictions and ground truths.')
parser.add_argument('predictions_dir',
                    help='Directory containing png files '
                         'representing segmentation predictions')
parser.add_argument('true_labels_dir',
                    help='Directory containing the corresponding true labels '
                         'of the images for which the predictions in '
                         '\'predictions_dir\' were made.')
parser.add_argument('log_output',
                    help='Location of CSV file which will be written to contain'
                         ' per-class IOU for each image.')
parser.add_argument('-t', '--threads', default=1, type=int,
                    help='Number of parallel threads to use')
parser.add_argument('-c', '--n_classes', default=3, type=int,
                    help='Number of classes in the data')


def mean_per_class_iou(pred_dir, truth_dir, n_classes):
    pred_paths = sorted(glob.glob(path_join(pred_dir, '*.png')) + glob.glob(
        path_join(pred_dir, '*.jpg')))
    truth_paths = sorted(glob.glob(path_join(truth_dir, '*.png')) + glob.glob(
        path_join(truth_dir, '*.jpg')))

    assert len(pred_paths) == len(truth_paths), \
        'Different number of images in prediction and truth directories'
    pd_paths = list(zip(pred_paths, truth_paths))
    assert np.all(
        [Path(x_p).stem == Path(y_p).stem for (x_p, y_p) in pd_paths]), \
        'Not all prediction and truth images are named correspondingly'

    iou_class_names = [f'iou_class_{i}' for i in range(args.n_classes)]
    iou_df = pd.DataFrame(columns=['truth_path', 'pred_path'] + iou_class_names)

    for pred_path, truth_path in pd_paths:
        pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
        truth_img = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE)

        resized_dim = (truth_img.shape[1], truth_img.shape[0])
        pred_img = cv2.resize(pred_img, resized_dim,
                              interpolation=cv2.INTER_NEAREST)
        pc_iou = per_class_iou(pred_img, truth_img, n_classes)
        iou_df = iou_df.append(
            pd.Series([truth_path, pred_path] + pc_iou, index=iou_df.columns),
            ignore_index=True)

        print(f'Per class IOU for {Path(pred_path).name}: {pc_iou}')

    return iou_df[iou_class_names].mean(axis=0, skipna=True), iou_df


# used in parallel computation
def mean_per_class_iou_single_image(pred_truth_paths):
    iou_class_names = [f'iou_class_{i}' for i in range(args.n_classes)]
    pred_path, truth_path, n_classes = pred_truth_paths

    pred_img = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
    truth_img = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE)

    resized_dim = (truth_img.shape[1], truth_img.shape[0])
    pred_img = cv2.resize(pred_img, resized_dim,
                          interpolation=cv2.INTER_NEAREST)

    pc_iou = per_class_iou(pred_img, truth_img, n_classes)
    print(f'Per class IOU for {Path(pred_path).name} '
          f'with size {pred_img.shape}: {pc_iou}')
    return pd.Series([truth_path, pred_path] + pc_iou,
                     index=['truth_path', 'pred_path'] + iou_class_names)


def mean_per_class_iou_parallel(pred_dir, truth_dir, n_threads, n_classes):
    pool = Pool(n_threads)
    pred_paths = sorted(glob.glob(path_join(pred_dir, '*.png')) + glob.glob(
        path_join(pred_dir, '*.jpg')))
    truth_paths = sorted(glob.glob(path_join(truth_dir, '*.png')) + glob.glob(
        path_join(truth_dir, '*.jpg')))

    assert len(pred_paths) == len(truth_paths), \
        'Different number of images in prediction and truth directories'
    pd_paths = list(
        zip(pred_paths, truth_paths, [n_classes] * len(truth_paths)))

    assert np.all(
        [Path(x_p).stem == Path(y_p).stem for (x_p, y_p, _) in pd_paths]), \
        'Not all prediction and truth images are named correspondingly'

    result_series = pool.map(mean_per_class_iou_single_image, pd_paths)
    iou_df = pd.DataFrame(result_series)
    iou_class_names = [f'iou_class_{i}' for i in range(args.n_classes)]
    mean_iou_for_datset = iou_df[iou_class_names].mean(axis=0, skipna=True)
    return mean_iou_for_datset, iou_df


if __name__ == '__main__':
    np.set_printoptions(precision=2)

    args = parser.parse_args()
    if args.threads == 1:
        pc_iou_, iou_df_ = mean_per_class_iou(args.predictions_dir,
                                              args.true_labels_dir,
                                              args.n_classes)
    else:
        pc_iou_, iou_df_ = mean_per_class_iou_parallel(args.predictions_dir,
                                                       args.true_labels_dir,
                                                       args.threads,
                                                       args.n_classes)

    print(pc_iou_)
    print(f'Mean IOU: {pc_iou_.mean()}')
    iou_df_.to_csv(args.log_output, index=False)
