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
from image_folder import MyImageFolder

use_gpu = torch.cuda.is_available()
if use_gpu:
    print("Using CUDA")

data_dir = '/Users/shikha/Documents/Fall2018/ComputerVision/Project/image_retrieval/dataset/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
batch_size = 3
out_features_size = 4096

# phase = TRAIN
# dataset_name = "shop_feature"
# image_dataset_name = "shop_feature_image"

# phase = TEST
# dataset_name = "consumer_feature"
# image_dataset_name = "consumer_feature_image"

# VGG-16 Takes 224x224 images as input, so we resize all of them
data_transforms = {
    TRAIN: transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    VAL: transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
    ]),
    TEST: transforms.Compose([
        transforms.Scale((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
}

image_datasets = {
    x: MyImageFolder(
        os.path.join(data_dir, x), 
        transform=data_transforms[x]
    )
    for x in [TRAIN, VAL, TEST]
}

dataloaders = {
    x: torch.utils.data.DataLoader(
        image_datasets[x], batch_size=batch_size,
        shuffle=True, num_workers=4
    )
    for x in [TRAIN, VAL, TEST]
}

dataset_sizes = {x: len(image_datasets[x]) for x in [TRAIN, VAL, TEST]}
print dataset_sizes[TRAIN]

for x in [TRAIN, VAL, TEST]:
    print("Loaded {} images under {}".format(dataset_sizes[x], x))
    
print("Classes: ")
class_names = image_datasets[TRAIN].classes
print(image_datasets[TRAIN].classes)

# Load the pretrained model from pytorch
model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
pre_model = torchvision.models.vgg19_bn(pretrained=True)
# model takes all the layers except the last classification layer
layers = list(pre_model.classifier.children())[0]
pre_model.classifier = nn.Sequential(*[layers])
# model.features = pre_model.features

for param in pre_model.parameters():
    param.requires_grad = False

def get_path_to_filename(paths):
    filenames = []
    for p in paths:
        f = p.split('/')[-1].split('.')[0]
        filenames.append(f)
    return filenames

# get the features as matrix for each image
# ------- SHOP -------
num_samples = dataset_sizes[phase]
result = np.zeros((num_samples, out_features_size))
result_filename = []
i = 0
for data in dataloaders[phase]:
    inputs, labels, paths = data
    filenames = get_path_to_filename(paths)
    result_filename += filenames
    inputs, labels = Variable(inputs), Variable(labels)
    outputs = pre_model(inputs)
    print outputs.shape
    start = i
    end = i+batch_size
    result[start:end, :] = outputs
    i += batch_size

file_name = data_dir+phase+'_feature.h5'
with h5py.File(file_name, 'w') as hf:
    hf.create_dataset(dataset_name, data=result)
    hf.create_dataset(image_dataset_name, data=result_filename)
