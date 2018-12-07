import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.models.vgg import model_urls
from torchvision import datasets, models, transforms

use_gpu = torch.cuda.is_available()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        model_urls['vgg19_bn'] = model_urls['vgg19_bn'].replace('https://', 'http://')
        pre_model = models.vgg19_bn(pretrained=True)
        # model takes all the layers except the last classification layer
        layers = list(pre_model.classifier.children())[0]
        # for param in pre_model.features.parameters():
        #     param.requires_grad = False
        pre_model.classifier = nn.Sequential(*[layers])
        # for param in pre_model.classifier.parameters():
        #   print param.requires_grad
        if use_gpu:
            pre_model = pre_model.cuda()
        self.model = pre_model

    def forward(self, anchor, positive=None, negative=None):
        anchor_output = self.model(anchor)
        positive_output = None
        negative_output = None
        if not positive is None:
            positive_output = self.model(positive)
        if not negative is None:
            negative_output = self.model(negative)
        
        return anchor_output, positive_output, negative_output


