#!/usr/bin/env python3
#
# Copyright, TU Dortmund 2020
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

import pandas as pd

from utils import path_join

parser = argparse.ArgumentParser(
    description='This script reads and analyzes the validation IoU files of '
                'either Deeplab or FCN networks, '
                'and orders the respective network checkpoints by mean IoU.')
parser.add_argument('val_results_root',
                    help='Root directory containing all checkpoint evaluations')
parser.add_argument('--output_file',
                    help='If set, this will contain the mIoU for each epoch.')
parser.add_argument('--epochs', type=int,
                    help='Number of epochs used in training of the model')
args = parser.parse_args()

mIOU = []

for i in list(range(0, args.epochs, 5)) + [args.epochs - 1]:
    iou_df_i_path = path_join(args.val_results_root, f'/ep{i}/iou.csv')
    iou_df_i = pd.read_csv(iou_df_i_path)
    mIOU_i = iou_df_i.drop(['truth_path', 'pred_path'], axis=1).mean(
        axis=0, skipna=True)
    mIOU_i['mIoU'] = mIOU_i.mean()
    mIOU_i['Epoch'] = i
    mIOU.append(mIOU_i)

mIOU = pd.DataFrame(mIOU)
mIOU.sort_values('mIoU', ascending=False, inplace=True)

if args.output_file is not None:
    mIOU.to_csv(args.output_file)

print(mIOU.head())
