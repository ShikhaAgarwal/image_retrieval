# from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torchvision.models.vgg import model_urls
import matplotlib.pyplot as plt
import time
import os
import copy

plt.ion()  

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/dataset'
TRAIN = 'train/'
VAL = 'val'
TEST = 'test'

print [os.path.join(data_dir, x) for x in [TRAIN, VAL, TEST]]
# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        # Data augmentation is a good practice for the train set
        # Here, we randomly crop the image to 224x224 and
        # randomly flip it horizontally. 
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ])
}

image_datasets = {
    x: datasets.ImageFolder(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=8,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}
print dataset_sizes

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

# Load the pretrained model from pytorch
model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
pre_model = torchvision.models.vgg19_bn(pretrained=True)
print pre_model
# model = nn.Sequential(*list(pre_model.children())[:-2])
# print model
model = pre_model.features

for param in model.parameters():    #----> 1
    param.requires_grad = False

inputs, labels = next(iter(dataloaders[TRAIN]))
inputs, labels = Variable(inputs), Variable(labels)
outputs = model(inputs)
print outputs