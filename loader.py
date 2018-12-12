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
import os
import copy
import h5py
from data_loader import data_loader
from deep_fashion_dataset import DeepFashionDataset

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
partition_file = 'dataset/list_eval_val_10.txt'

TRAIN = 'train'
VAL = 'val'
TEST = 'test'
data_types = [VAL]
mode = 'shop'
batch_size = 16
out_features_size = 4096

file_name = data_dir + "baseline_" + data_types[0] + '_' + mode + '_feature.h5'
dataset_name = data_types[0] + '_features'
image_dataset_name = data_types[0] + '_image_names'
phase = data_types[0]

dataloaders, num_samples = data_loader(data_dir, partition_file, data_types, batch_size, shuffle=False, mode=mode)

# Load the pretrained model from pytorch
model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
pre_model = torchvision.models.vgg19_bn(pretrained=True)
# model takes all the layers except the last classification layer
print pre_model
layers = list(pre_model.classifier.children())[0]
pre_model.classifier = nn.Sequential(*[layers])
# model.features = pre_model.features

for param in pre_model.parameters():
    param.requires_grad = False

# get the features as matrix for each image
# ------- SHOP -------
result = np.zeros((num_samples, out_features_size))
result_filename = []
i = 0
for data in dataloaders[phase]:
    anchor_list, _, _, target, paths_list = data
    #print paths_list
    #filenames = get_path_to_filename(paths)
    count = 0
    anchor = []
    for a, p in zip(anchor_list, paths_list):
        if p in result_filename:
            continue
        anchor.append(a)
        result_filename.append(p)
        count += 1
    if count == 0:
        continue
    result_img = torch.stack(anchor)
    #result_filename += paths
    if use_gpu:
        anchor = Variable(result_img.cuda())
    else:
        anchor = Variable(result_img)
    anchor_output, _, _ = vgg_model(anchor)
    start = i
    end = i + count
    result[start:end, :] = anchor_output.data
    i += count
 
print i , "last i"
 
with h5py.File(file_name, 'w') as hf:
    hf.create_dataset(dataset_name, data=result[:i,:])
    hf.create_dataset(image_dataset_name, data=result_filename)
