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


data_dir = '/home/sneha/image_retrieval/dataset/'
model_dir = '/home/sneha/saved_model/'
TRAIN = 'train'
VAL = 'val'
TEST = 'test'
batch_size = 1

phase = TRAIN

num_epochs = 2
margin = 1.
use_gpu = torch.cuda.is_available()
criterion = TripletLoss(margin)
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
    x: DeepFashionDataset(
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


vgg_features = model.FeatureExtractor()
#triplet_net = model.TripletNet(vgg_features)

optimizer = optim.Adam(vgg_features.model.classifier.parameters(), lr=0.001)

for i in range(num_epochs):
    for data in dataloaders[phase]:
    	anchor, pos_sample, neg_sample, target, path = data
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
        print "loss=", loss
    torch.save(vgg_features.state_dict(), model_dir+"_"+str(i)+".weights")
