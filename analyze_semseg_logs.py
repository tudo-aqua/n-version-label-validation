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
import re

import pandas as pd

parser = argparse.ArgumentParser(
    description='This script reads the training script of a semseg '
                'training run that evaluated its model on the '
                'validation dataset between epochs, and finds the epoch '
                'with the highest mIoU among them.')
parser.add_argument('semseg_logfile',
                    help='Log file written during semseg training run.')
parser.add_argument('--output_file',
                    help='If set, this will contain the mIoU/mAcc/allAcc values'
                         ' for each epoch.')


# Extract float list of mIoU, mAcc, allAcc from log line
def extract_values(line):
    front_cleared = re.sub(r'\[.*\] Val result: mIoU/mAcc/allAcc ', '', line)
    numbers_as_string = re.sub(r'\.\n', '', front_cleared)
    numbers_as_string = numbers_as_string.split('/')
    return list(map(float, numbers_as_string))


args = parser.parse_args()

with open(args.semseg_logfile, 'r') as f:
    log_contents = f.readlines()

eval_lines = [line for line in log_contents if 'Val result' in line]
if len(eval_lines) == 0:
    raise ValueError('No validation set results found in the given file. '
                     'Possibly the wrong file was chosen, or the'
                     '\'evaluate\' attribute was not set '
                     'during the training run.')

metrics = list(map(extract_values, eval_lines))

metrics_df = pd.DataFrame(metrics, columns=['mIoU', 'mAcc', 'allAcc'])
metrics_df['Epoch'] = range(1, metrics_df.shape[0] + 1)
metrics_df.sort_values('mIoU', ascending=False, inplace=True)

if args.output_file is not None:
    metrics_df.to_csv(args.output_file)

print(metrics_df.head())
