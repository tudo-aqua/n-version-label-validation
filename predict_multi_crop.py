"""
The following code was slightly adapted from its original implementation at
https://github.com/hszhao/semseg/blob/master/tool/test.py

MIT License

Copyright (c) 2019 Hengshuang Zhao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the 'Software'), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import logging
import os
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn.parallel
import torch.utils.data

from semseg.util.util import AverageMeter, check_makedirs, colorize

cv2.ocl.setUseOpenCL(False)


def get_logger():
    logger_name = 'main-logger'
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    fmt = '[%(asctime)s %(levelname)s %(filename)s line ' \
          '%(lineno)d %(process)d] %(message)s'
    handler.setFormatter(logging.Formatter(fmt))
    logger.addHandler(handler)
    return logger


def net_process(model, image, flip=True):
    input_ = torch.from_numpy(image.transpose((2, 0, 1))).float()
    input_ = input_.unsqueeze(0).cuda()
    if flip:
        input_ = torch.cat([input_, input_.flip(3)], 0)
    with torch.no_grad():
        output = model(input_)['out']
    _, _, h_i, w_i = input_.shape
    _, _, h_o, w_o = output.shape
    if (h_o != h_i) or (w_o != w_i):
        output = F.interpolate(output, (h_i, w_i), mode='bilinear',
                               align_corners=True)
    output = F.softmax(output, dim=1)
    if flip:
        output = (output[0] + output[1].flip(2)) / 2
    else:
        output = output[0]
    output = output.data.cpu().numpy()
    output = output.transpose(1, 2, 0)
    return output


def scale_process(model, image, classes, crop_h, crop_w, h, w, mean,
                  stride_rate=2 / 3):
    ori_h, ori_w, _ = image.shape
    pad_h = max(crop_h - ori_h, 0)
    pad_w = max(crop_w - ori_w, 0)
    pad_h_half = int(pad_h / 2)
    pad_w_half = int(pad_w / 2)
    if pad_h > 0 or pad_w > 0:
        image = cv2.copyMakeBorder(image, pad_h_half, pad_h - pad_h_half,
                                   pad_w_half, pad_w - pad_w_half,
                                   cv2.BORDER_CONSTANT, value=mean)
    new_h, new_w, _ = image.shape
    stride_h = int(np.ceil(crop_h * stride_rate))
    stride_w = int(np.ceil(crop_w * stride_rate))
    grid_h = int(np.ceil(float(new_h - crop_h) / stride_h) + 1)
    grid_w = int(np.ceil(float(new_w - crop_w) / stride_w) + 1)
    prediction_crop = np.zeros((new_h, new_w, classes), dtype=float)
    count_crop = np.zeros((new_h, new_w), dtype=float)
    for index_h in range(0, grid_h):
        for index_w in range(0, grid_w):
            s_h = index_h * stride_h
            e_h = min(s_h + crop_h, new_h)
            s_h = e_h - crop_h
            s_w = index_w * stride_w
            e_w = min(s_w + crop_w, new_w)
            s_w = e_w - crop_w
            image_crop = image[s_h:e_h, s_w:e_w].copy()
            count_crop[s_h:e_h, s_w:e_w] += 1
            tmp = net_process(model, image_crop)
            prediction_crop[s_h:e_h, s_w:e_w, :] += tmp
    prediction_crop /= np.expand_dims(count_crop, 2)
    prediction_crop = prediction_crop[
                      pad_h_half:pad_h_half + ori_h,
                      pad_w_half:pad_w_half + ori_w]
    prediction = cv2.resize(prediction_crop, (w, h),
                            interpolation=cv2.INTER_LINEAR)
    return prediction


def predict_multi_crop(logger, test_loader, data_list, model, classes, mean,
                       base_size, crop_h, crop_w, scales,
                       gray_folder,
                       color_folder, colors):
    logger.info('\n >>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
    data_time = AverageMeter()
    batch_time = AverageMeter()
    model.eval()
    end = time.time()
    for i, (input_, _) in enumerate(test_loader):
        data_time.update(time.time() - end)
        input_ = np.squeeze(input_.numpy(), axis=0)
        image = np.transpose(input_, (1, 2, 0))
        h, w, _ = image.shape
        prediction = np.zeros((h, w, classes), dtype=float)
        for scale in scales:
            long_size = round(scale * base_size)
            new_h = long_size
            new_w = long_size
            if h > w:
                new_w = round(long_size / float(h) * w)
            else:
                new_h = round(long_size / float(w) * h)
            image_scale = cv2.resize(image, (new_w, new_h),
                                     interpolation=cv2.INTER_LINEAR)
            prediction += scale_process(model, image_scale, classes, crop_h,
                                        crop_w, h, w, mean)
        prediction /= len(scales)
        prediction = np.argmax(prediction, axis=2)
        batch_time.update(time.time() - end)
        end = time.time()
        if ((i + 1) % 10 == 0) or (i + 1 == len(test_loader)):
            logger.info('Test: [{}/{}] '
                        'Data {data_time.val:.3f} ({data_time.avg:.3f}) '
                        'Batch {batch_time.val:.3f} ({batch_time.avg:.3f}).'.
                        format(i + 1, len(test_loader), data_time=data_time,
                               batch_time=batch_time))
        check_makedirs(gray_folder)
        check_makedirs(color_folder)
        gray = np.uint8(prediction)
        color = colorize(gray, colors)
        image_path, _ = data_list[i]
        image_name = image_path.split('/')[-1].split('.')[0]
        gray_path = os.path.join(gray_folder, image_name + '.png')
        color_path = os.path.join(color_folder, image_name + '.png')
        cv2.imwrite(gray_path, gray)
        color.save(color_path)
    logger.info('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')
