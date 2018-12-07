import model
from data_loader import data_loader
import torch
import os
import numpy as np
from torch.autograd import Variable
import h5py

TRAIN = 'train'
VAL = 'val'
TEST = 'test'

data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/'
model_dir = '/mnt/nfs/scratch1/shikhaagarwa/saved_model/'
#model_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/output_weights/'
path = os.path.join(model_dir, '_15.weights')
batch_size = 16
data_types = [VAL]
out_features_size = 4096

dataset_name = data_types[0] + '_features'
image_dataset_name = data_types[0] + '_image_names'
phase = data_types[0]
use_gpu = torch.cuda.is_available() 
print "gpu = ", use_gpu

def get_path_to_filename(paths):
    filenames = []
    for p in paths:
        f = p.split('/')[-1].split('.')[0]
        filenames.append(f)
    return filenames

vgg_model = model.FeatureExtractor()
vgg_model.load_state_dict(torch.load(path))

dataloaders, num_samples = data_loader(data_dir, data_types, batch_size,mode=data_types[0])
result = np.zeros((num_samples, out_features_size))
result_filename = []
i = 0
for data in dataloaders[phase]:
    anchor, _, _, target, path = data
    filenames = get_path_to_filename(paths)
    result_filename += filenames
    if use_gpu:
        anchor = Variable(anchor.cuda())
    else:
        anchor = Variable(anchor)
    anchor_output, _, _ = vgg_model(anchor)
    start = i
    end = i+batch_size
    result[start:end, :] = anchor_output.data
    i += batch_size

file_name = data_dir + data_types[0] + '_feature.h5'
with h5py.File(file_name, 'w') as hf:
    hf.create_dataset(dataset_name, data=result)
    hf.create_dataset(image_dataset_name, data=result_filename)
