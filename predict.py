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
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from tqdm import tqdm

from dataset import get_mean_std
from dataset import get_val_loader
from predict_multi_crop import predict_multi_crop, get_logger
from semseg.util import config
from semseg.util.util import colorize


def predict(config_file, model_file, output_dir, n_classes, architecture,
            multi_crop=False, aux_loss=None):
    pred_dir = Path(output_dir) / 'predictions'
    col_dir = Path(output_dir) / 'colorized'
    pred_dir.mkdir(exist_ok=True)
    col_dir.mkdir(exist_ok=True)

    if architecture == 'deeplab':
        model = deeplabv3_resnet101(pretrained=False, progress=True,
                                    num_classes=n_classes, aux_loss=aux_loss)
    elif architecture == 'fcn':
        model = fcn_resnet101(pretrained=False, progress=True,
                              num_classes=n_classes, aux_loss=aux_loss)
    else:
        raise ValueError('Unknown architecture specified. '
                         'Please choose either \'fcn\' or \'deeplab\'.')

    model.load_state_dict(torch.load(model_file))
    model.eval()
    model.cuda()
    criterion = CrossEntropyLoss(ignore_index=255)

    test_loader, data_list = get_val_loader(config_file)
    data_loader = tqdm(test_loader, total=len(test_loader), ncols=80)

    color_palette = np.loadtxt(
        'semseg/data/cityscapes/cityscapes_colors.txt').astype('uint8')

    if not multi_crop:

        for i, (img, label) in enumerate(data_loader):
            img_path, label_path = data_list[i]
            img = torch.cuda.FloatTensor(img.numpy())
            label = torch.cuda.FloatTensor(label.numpy()).long()

            output = model(img)['out']
            loss = criterion(output, label)
            data_loader.set_postfix(loss=loss.item())

            pred_path = pred_dir / Path(img_path).name
            col_path = col_dir / Path(img_path).name

            # Apply argmax to prediction image. The resulting indices
            # represent an 8-bit image with each pixel value corresponding
            # to the respective class label
            values, indices = torch.max(torch.squeeze(output), 0)
            pred = indices.cpu().numpy()

            color_pred = colorize(pred, color_palette)

            cv2.imwrite(str(pred_path), pred)
            color_pred.save(str(col_path))

    else:
        test_args = config.load_cfg_from_cfg_file(config_file)
        mean, std = get_mean_std(test_args.mean_std)
        crop_h = test_args.train_h
        crop_w = test_args.train_w

        logger = get_logger()

        predict_multi_crop(logger, test_loader, data_list, model, n_classes,
                           mean, test_args.base_size,
                           crop_h, crop_w, test_args.scales, pred_dir, col_dir,
                           color_palette)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='This script lets the neural network predict segmentations '
                    'for the validation dataset, and saves the colorized and '
                    'non-colorized predictions in the output directory.')
    parser.add_argument('model_file',
                        help='Path to a model file of the architecture '
                             'given with --architecture.')
    parser.add_argument('val_dir',
                        help='Directory containing png files of images and '
                             'corresponding labels. The contained directory '
                             'structure is supposed to be ./{images,labels}.')
    parser.add_argument('output_dir',
                        help='Directory to which predicted segmentations '
                             'are written.')
    parser.add_argument('config', help='YAML config file to use')
    parser.add_argument('-c', '--n_classes', type=int, default=3,
                        help='Number of classes in the data, defaults to 3.')
    parser.add_argument('-a', '--architecture', choices=['deeplab', 'fcn'],
                        required=True,
                        help='Neural network architecture to be used for '
                             'training. Can either be \'deeplab\' or \'fcn\' '
                             'to choose DeeplabV3_Resnet101 or FCN_Resnet101.')
    parser.add_argument('-m', '--multi_crop', action='store_true',
                        help='Use a prediction aggregated over multiple image '
                             'crops, and possibly multiple scales. '
                             'If not set, the whole image will be fed into '
                             'the model at once.')
    args = parser.parse_args()

    predict(args.config, args.model_file, args.output_dir, args.n_classes,
            args.architecture,
            multi_crop=args.multi_crop, aux_loss=True)
