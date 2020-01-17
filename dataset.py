"""
The following code was adapted from its original implementation at:
https://github.com/hszhao/semseg/blob/master/tool/train.py

The respective parent project is located at: https://github.com/hszhao/semseg

MIT License

Copyright (c) 2019 Hengshuang Zhao
Copyright of Modifications TU Dortmund 2020, Sebastian Gerard

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

import torch

from semseg.util import dataset, transform, config


def get_mean_std(data_set):
    """Get mean and standard deviation of RGB values for the given dataset."""
    value_scale = 255
    mean_cs = [0.485, 0.456, 0.406]
    mean_cs = [item * value_scale for item in mean_cs]
    std_cs = [0.229, 0.224, 0.225]
    std_cs = [item * value_scale for item in std_cs]

    mean_bdd = [0.3132649, 0.31416275, 0.29762544]
    mean_bdd = [item * value_scale for item in mean_bdd]
    std_bdd = [0.4244708279394972, 0.4158931970801661, 0.39015395151921883]
    std_bdd = [item * value_scale for item in std_bdd]

    if data_set == 'cs':
        mean = mean_cs
        std = std_cs
    elif data_set == 'bdd':
        mean = mean_bdd
        std = std_bdd
    else:
        raise ValueError(f'args.mean_std value not supported: {data_set}')
    return mean, std


def get_train_loader(config_file='cityscapes_deeplab.yaml'):
    args = config.load_cfg_from_cfg_file(config_file)
    mean, std = get_mean_std(args.mean_std)

    train_transform = transform.Compose([
        transform.RandScale([args.scale_min, args.scale_max]),
        transform.RandRotate([args.rotate_min, args.rotate_max], padding=mean,
                             ignore_label=args.ignore_label),
        transform.RandomGaussianBlur(),
        transform.RandomHorizontalFlip(),
        transform.Crop([args.train_h, args.train_w], crop_type='rand',
                       padding=mean, ignore_label=args.ignore_label),
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)
    ])

    train_data = dataset.SemData(split='train', data_root=args.data_root,
                                 data_list=args.train_list,
                                 transform=train_transform)

    train_loader = torch.utils.data.DataLoader(train_data,
                                               batch_size=args.batch_size,
                                               shuffle=True,
                                               num_workers=args.workers,
                                               pin_memory=True,
                                               sampler=None, drop_last=True)

    return args, train_loader


def get_val_loader(config_file='cityscapes_deeplab.yaml'):
    args = config.load_cfg_from_cfg_file(config_file)
    mean, std = get_mean_std(args.mean_std)

    val_transform = transform.Compose([
        transform.ToTensor(),
        transform.Normalize(mean=mean, std=std)])

    val_data = dataset.SemData(split='val',
                               data_root=args.data_root,
                               data_list=args.val_list,
                               transform=val_transform)

    val_loader = torch.utils.data.DataLoader(val_data,
                                             batch_size=1,
                                             shuffle=False,
                                             num_workers=args.workers,
                                             pin_memory=True)

    return val_loader, val_data.data_list
