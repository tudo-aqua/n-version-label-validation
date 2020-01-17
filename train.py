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

import pandas as pd
import torch
import torch.optim as optim
from torch.backends import cudnn
from torch.nn import CrossEntropyLoss
from torchvision.models.segmentation import deeplabv3_resnet101, fcn_resnet101
from tqdm import tqdm

from dataset import get_train_loader
from predict import predict
from utils import path_join

aux_loss = True

parser = argparse.ArgumentParser(
    description='Training deeplab or fcn on cityscapes dataset')
parser.add_argument('--config', type=str, default='cityscapes_deeplab.yaml',
                    help='config file')
args0 = parser.parse_args()

cudnn.benchmark = True
args, data_loader = get_train_loader(args0.config)

if args.architecture == 'deeplab':
    model = deeplabv3_resnet101(pretrained=False, progress=True,
                                num_classes=args.classes, aux_loss=aux_loss)
elif args.architecture == 'fcn':
    model = fcn_resnet101(pretrained=False, progress=True,
                          num_classes=args.classes, aux_loss=aux_loss)
else:
    raise ValueError('Unknown architecture specified. '
                     'Please choose either \'fcn\' or \'deeplab\'.')

model.cuda()
print(model)
criterion = CrossEntropyLoss(ignore_index=args.ignore_label)
optimizer = optim.SGD(model.parameters(), lr=args.base_lr,
                      momentum=args.momentum)

loss_log = []

for epoch in range(args.epochs):

    running_loss_ep = 0.0

    # Use tqdm progress bar for iterations
    data_generator = tqdm(enumerate(data_loader, 0), total=len(data_loader),
                          ncols=80)
    for i, (inputs, labels) in data_generator:
        # For some reason .cuda() did not work,
        # so this is how we move the data to the GPU
        inputs = torch.cuda.FloatTensor(inputs.numpy())
        labels = torch.cuda.FloatTensor(labels.numpy()).long()
        optimizer.zero_grad()

        # forward + backward + optimize
        outputs = model(inputs)
        loss_regular = criterion(outputs['out'], labels)
        loss_aux = criterion(outputs['aux'], labels)
        loss = loss_regular + args.aux_weight * loss_aux
        loss.backward()
        optimizer.step()

        loss_val = loss.item()
        running_loss_ep += loss_val

        # print statistics
        data_generator.set_description(
            'Epoch {}/{}'.format(epoch + 1, args.epochs))
        data_generator.set_postfix(loss=loss_val)

    loss_log.append([epoch, (running_loss_ep / len(data_loader))])

    if epoch % 5 == 0:
        torch.save(model.state_dict(),
                   path_join(args.output_dir, f'model_ep{epoch}.dict'))

final_model_path = path_join(args.output_dir, f'model_ep{args.epochs - 1}.dict')
torch.save(model.state_dict(), final_model_path)
loss_df = pd.DataFrame(data=loss_log, columns=['epoch', 'cross_entropy_loss'])
loss_df.to_csv(path_join(args.output_dir, f'train_loss.csv'), index=None)
print('Finished Training')

print('Starting Evaluation')
predict(args0.config, final_model_path, args.output_dir, args.classes,
        args.architecture, aux_loss=aux_loss)
