import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import model
from torchvision import datasets, models, transforms
import os 
from deep_fashion_dataset import DeepFashionDataset
from loss import TripletLoss
import torch.optim as optim
from data_loader import data_loader

data_dir = '/mnt/nfs/scratch1/snehabhattac/vision_data/processed_data/'
model_dir = '/mnt/nfs/scratch1/snehabhattac/saved_model/run3/'
partition_file = 'dataset/list_eval_partition_train.txt'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
batch_size = 8
phase = TRAIN
data_types = [phase]
num_epochs = 100
margin = 0.3
learning_rate = 0.001
use_gpu = torch.cuda.is_available()

criterion = TripletLoss(margin)
vgg_features = model.FeatureExtractor()
dataloaders, _ = data_loader(data_dir, partition_file, data_types, batch_size)
optimizer = optim.Adam(vgg_features.model.parameters(), lr=learning_rate)

for i in range(num_epochs):
    print "starting epoch", i
    loss_list = []
    for data in dataloaders[phase]:
        anchor, pos_sample, neg_sample, target, path = data
        #print anchor.shape
        if use_gpu:
            anchor = Variable(anchor.cuda())
            pos_sample = Variable(pos_sample.cuda())
            neg_sample = Variable(neg_sample.cuda())
        else:
            anchor = Variable(anchor)
            pos_sample = Variable(pos_sample)
            neg_sample = Variable(neg_sample)
        anchor_output, positive_output, negative_output = vgg_features(anchor, pos_sample, neg_sample)
        optimizer.zero_grad()   
        loss = criterion(anchor_output, positive_output, negative_output)
        loss.backward()
        optimizer.step()  
        loss_list.append(loss.item())
    avg_loss = sum(loss_list) / float(len(loss_list))
    print "epoch = ", i, ", loss = ", avg_loss
    if (i+1) % 5 == 0:
        torch.save(vgg_features.model.state_dict(), model_dir+"_"+str(i+1)+".weights")
