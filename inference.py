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

data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/processed_data/'
model_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/saved_model/run_split_20/'
#model_dir = '/mnt/nfs/scratch1/shikhaagarwa/saved_model/run_10k_16/'

partition_file = 'dataset/list_eval_train_1k.txt'
#model_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/output_weights/'
path = os.path.join(model_dir, '_50.weights')
data_types = [TRAIN]
mode = 'comsumer'
batch_size = 16
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
vgg_model.model.load_state_dict(torch.load(path))
#vgg_model.model.eval()

dataloaders, num_samples = data_loader(data_dir, partition_file, data_types, batch_size, shuffle=False, mode=mode)
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
file_name = data_dir + data_types[0] + '_' + mode + '_feature.h5'
with h5py.File(file_name, 'w') as hf:
    hf.create_dataset(dataset_name, data=result[:i,:])
    hf.create_dataset(image_dataset_name, data=result_filename)
