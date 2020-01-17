"""
The following code was adapted from its original implementation by ycszen at:
https://github.com/ycszen/TorchSeg/blob/6fe74794903f58355e099504e697fd46763a4a8e/furnace/seg_opr/metric.py

The respective parent project is located at: https://github.com/ycszen/TorchSeg

MIT License

Copyright (c) 2019 ycszen

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

import numpy as np


def per_class_iou(pred, truth, n_classes):
    assert (pred.shape == truth.shape)
    pred = pred.flatten()
    truth = truth.flatten()

    # Compute confusion matrix
    conf_mat = np.bincount(n_classes * truth.astype(int) + pred.astype(int),
                           minlength=n_classes ** 2)
    conf_mat = conf_mat.reshape(n_classes, n_classes)

    # Compute per-class intersection over union metric from confusion matrix
    iou_per_class = np.diag(conf_mat) / (
            conf_mat.sum(1) + conf_mat.sum(0) - np.diag(conf_mat))

    return iou_per_class.tolist()
