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
from matplotlib import cm

from utils import path_join

TRUTH_PATHS = []


def setup_args_parser():
    parser = argparse.ArgumentParser(
        description='This script takes several folders of different predictions'
                    'for the same ground truth and'
                    'computes the majority voted class per pixel.'
                    'For each predicted image it computes: '
                    '   a) An image containing the majority classification'
                    '      for each pixel '
                    '   b) An image containing the number of votes'
                    '      that the majority class received per pixel'
                    '   c) An image showing for each pixel by how many votes'
                    '      the majority class disagreed with the ground truth '
                    '   d) An image that visualizes these aspects in a format'
                    '      that can easily be visually inspected, unlike'
                    '      the previously mentioned images. '
                    'It computes metrics based on the majority voting results,'
                    'which can be used to categorize the images with'
                    'majority_vote_sorting.py and are written'
                    'to the output_dir.')
    help_td = ('Directory containing the ground truth labels of the images '
               'for which the predictions were made')
    parser.add_argument('truth_dir', help=help_td)

    help_color_img_d = ('Directory containing the color images'
                        'for which predictions were made'
                        '(used for visualization).')
    parser.add_argument('color_img_dir', help=help_color_img_d)

    help_od = ('Directory to which the majority vote metrics should be written'
               '(and majority voted images if -w is set)')
    parser.add_argument('output_dir', help=help_od)

    help_pd = ('Variable number of directories containing png files '
               'representing segmentation predictions for the same pictures, '
               'named exactly the same as the respective ground truth labels')
    parser.add_argument('prediction_dirs', nargs='+', help=help_pd)

    parser.add_argument('-t', '--threads', type=int,
                        default=1,
                        help='Number of threads to use for computation')
    return parser


def majority_vote_pixelwise(predicted_classes):
    """Compute the majoriy votes per pixel.

    Input:  class predictions of all models at a single pixel
    Output: class which received the most votes and
            the number of votes that it received
    """
    values, counts = np.unique(predicted_classes, return_counts=True)
    max_idx = np.argmax(counts)
    majority_class = values[max_idx]
    n_votes = counts[max_idx]
    return np.array([majority_class, n_votes])


def swap_lane_classes(x):
    if x == 1:
        return 2
    if x == 2:
        return 1
    return x


def highest_horizontal_rectangle(arr, limit=0.9):
    """Compute highest rectangle.

    Input:  Image containing majority vote classification
    Output: Length of the longest contiguous set of image rows
            for which at least (limit*img_width) pixels
            were classified as belonging to the ego lane.
    """
    previous_was_rectangle = False
    truth_list = np.sum(arr == 1, axis=1) >= (limit * arr.shape[1])

    best_solution = {'length': 0, 'start': 0, 'end': 0}
    current_solution = {'length': 0, 'start': 0, 'end': 0}

    for j, value in enumerate(truth_list):
        if value:
            if previous_was_rectangle:
                current_solution['end'] += 1
            else:
                current_solution = {'length': 0, 'start': j, 'end': j}
                previous_was_rectangle = True
        if not value or j == len(truth_list) - 1:
            if previous_was_rectangle:
                current_solution['length'] = (current_solution['end'] -
                                              current_solution['start'] + 1)
                previous_was_rectangle = False
                if current_solution['length'] > best_solution['length']:
                    best_solution = current_solution
    return best_solution['length'] / arr.shape[0]


def load_prediction_imgs_from_tuple(truth_path, prediction_dirs):
    pred_imgs = []
    img_name = Path(truth_path).name
    for pred_dir in prediction_dirs:
        pred_img_path = pred_dir + '/' + img_name
        pred_img = cv2.imread(pred_img_path, cv2.IMREAD_GRAYSCALE)
        print(f'{pred_img_path}: {pred_img.shape}')
        pred_imgs.append(pred_img)

    return np.stack(pred_imgs, axis=-1)


def write_images(output_dir, truth_path, color_img_path, visual_path,
                 maj_voted, strength_of_disagreement_img, equality_img):
    img_width = equality_img.shape[1]
    img_height = equality_img.shape[0]

    # Write already computed images
    cv2.imwrite(output_dir + '/majority_voted/' +
                Path(truth_path).name, maj_voted[:, :, 0])
    cv2.imwrite(output_dir + '/votes/' +
                Path(truth_path).name, maj_voted[:, :, 1])
    cv2.imwrite(output_dir + '/disagreement_strength/' +
                Path(truth_path).name, strength_of_disagreement_img)

    # Create image for direct visual inspection:
    pad_top = 40
    pad_middle = 10

    color_map = [i * (255 // 3) for i in range(4)]
    colorize = np.vectorize(lambda x: color_map[x])

    imgs = [equality_img * 255,
            colorize(maj_voted[:, :, 1]),
            colorize(np.multiply(np.invert(equality_img), maj_voted[:, :, 1])),
            colorize(maj_voted[:, :, 0])]

    img_captions = ['Majority vote == ground truth?',
                    'Number of Votes',
                    'Disagreement with ground truth',
                    'Majority-predicted classes']

    # Combine all four images on a single canvas with respective captions
    canvas = np.full(shape=(img_height + pad_top,
                            img_width * 5 + 4 * pad_middle),
                     fill_value=23, dtype=np.uint8)
    for j, (img, img_caption) in enumerate(zip(imgs, img_captions)):
        left = (img_width + pad_middle) * j
        right = (img_width + pad_middle) * j + img_width
        canvas[pad_top:, left:right] = img.astype(np.uint8)
        canvas = cv2.putText(img=canvas,
                             text=img_caption,
                             org=(left, pad_top - 10),
                             fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                             fontScale=0.4,
                             color=120,
                             thickness=1)
        if type(canvas) == cv2.UMat:
            canvas = cv2.UMat.get(canvas)

    canvas = cm.hot(canvas, bytes=True)[:, :, :3]
    canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    # Add color image
    left = (img_width + pad_middle) * 4
    right = (img_width + pad_middle) * 4 + img_width
    canvas[pad_top:, left:right] = cv2.imread(color_img_path)

    cv2.imwrite(visual_path, canvas)


def compute_metrics(truth_path,
                    visual_path,
                    color_img_path,
                    truth_img,
                    equality_img,
                    strength_of_disagreement_img,
                    maj_vote_result,
                    n_preds):
    """Computes several different metrics.

    Not all of them were used in the paper."""
    img_width = equality_img.shape[1]
    img_height = equality_img.shape[0]

    vswap_lanes = np.vectorize(swap_lane_classes)

    n_disagreeing_pixels = img_height * img_width - np.sum(equality_img)
    disagreeing_pixels_ratio = n_disagreeing_pixels / (img_height * img_width)
    sum_disagreeing_votes = np.sum(strength_of_disagreement_img)
    if n_disagreeing_pixels == 0:
        mean_disagreeing_vote = 0
    else:
        mean_disagreeing_vote = sum_disagreeing_votes / n_disagreeing_pixels
    mean_divergent_votes = np.mean(n_preds - maj_vote_result[:, :, 1])
    votes_standard_deviation = np.std(maj_vote_result[:, :, 1])
    bg_ratio_mv = np.mean(maj_vote_result[:, :, 0] == 0)
    bg_ratio_original = np.mean(truth_img == 0)

    lanes_merged_disagreement = np.mean(
        (maj_vote_result[:, :, 0] > 0) != (truth_img > 0))

    swapped_lanes_disagreement = np.mean(
        vswap_lanes(maj_vote_result[:, :, 0]) != truth_img)

    merged_swapped_diff = (swapped_lanes_disagreement -
                           lanes_merged_disagreement)

    horizontal_rectangle = highest_horizontal_rectangle(
        maj_vote_result[:, :, 0], limit=0.9)

    return pd.Series({'Ground truth': truth_path,
                      'Visualization': visual_path,
                      'Color image': color_img_path,
                      'Ratio of background pixels in majority vote':
                          bg_ratio_mv,
                      'Ratio of background pixels in ground truth':
                          bg_ratio_original,
                      'Ratio of disagreeing pixels': disagreeing_pixels_ratio,
                      'Sum of disagreeing votes': sum_disagreeing_votes,
                      'Mean disagreeing votes': mean_disagreeing_vote,
                      'Mean divergent votes': mean_divergent_votes,
                      'Votes standard deviation': votes_standard_deviation,
                      'Disagreement with merged lane classes':
                          lanes_merged_disagreement,
                      'Swapped lanes disagreement': swapped_lanes_disagreement,
                      'Diff btw swapped and merged': merged_swapped_diff,
                      'Height of horizontal rectangle': horizontal_rectangle})


def compute_metrics_for_single_image(truth_path,
                                     color_img_dir,
                                     prediction_dirs,
                                     output_dir, n_preds):
    truth_img = cv2.imread(truth_path, cv2.IMREAD_GRAYSCALE)
    color_img_path = path_join(color_img_dir, Path(truth_path).name)
    print(truth_path)

    # Load all predictions pertaining to the same image and stack them,
    # taking each prediction as a channel
    pred_imgs = load_prediction_imgs_from_tuple(truth_path, prediction_dirs)

    maj_vote_result = np.apply_along_axis(
        majority_vote_pixelwise, -1, pred_imgs)

    # Compare majority vote classification with ground truth
    equality_img = np.equal(truth_img, maj_vote_result[:, :, 0])

    # Compute strength of disagreement between
    # majority vote classification and ground truth
    strength_of_disagreement_img = np.multiply(np.invert(equality_img),
                                               maj_vote_result[:, :, 1])

    visual_path = path_join(output_dir, 'visual', Path(truth_path).name)
    write_images(output_dir, truth_path, color_img_path, visual_path,
                 maj_vote_result, strength_of_disagreement_img, equality_img)

    return compute_metrics(
        truth_path,
        visual_path,
        color_img_path,
        truth_img,
        equality_img,
        strength_of_disagreement_img, maj_vote_result,
        n_preds)


def compute_metrics_distr(args, n_preds):
    pool = Pool(processes=args.threads)
    pred_dirs = tuple(args.prediction_dirs)
    results = [pool.apply_async(compute_metrics_for_single_image,
                                (truth_path,
                                 args.color_img_dir,
                                 pred_dirs,
                                 args.output_dir,
                                 n_preds)) for truth_path in TRUTH_PATHS]
    return [res.get() for res in results]


def main():
    global TRUTH_PATHS
    args = setup_args_parser().parse_args()

    if len(args.prediction_dirs) < 2:
        raise ValueError(
            'Please specify at least two directories containing predictions.')

    n_preds = len(args.prediction_dirs)
    TRUTH_PATHS = sorted(glob.glob(path_join(args.truth_dir, '*.png')))

    (Path(args.output_dir) / 'majority_voted').mkdir(
        parents=True, exist_ok=True)
    (Path(args.output_dir) / 'votes').mkdir(
        parents=True, exist_ok=True)
    (Path(args.output_dir) / 'visual').mkdir(
        parents=True, exist_ok=True)
    (Path(args.output_dir) / 'disagreement_strength').mkdir(
        parents=True, exist_ok=True)

    if args.threads < 1:
        raise ValueError('Number of threads must be at least 1.')
    if args.threads == 1:
        metrics = [compute_metrics_for_single_image(truth_path,
                                                    args.color_img_dir,
                                                    args.prediction_dirs,
                                                    args.output_dir,
                                                    n_preds)
                   for truth_path in TRUTH_PATHS]
    else:
        metrics = compute_metrics_distr(args, n_preds)

    pd.DataFrame(metrics).to_csv(path_join(args.output_dir, 'metrics.csv'),
                                 index=None)


if __name__ == '__main__':
    main()
