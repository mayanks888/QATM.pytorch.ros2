from pathlib import Path
from seaborn import color_palette
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models, transforms, utils
import copy
from utils import *
from PIL import Image

# load the original Alexnet model
model = models.vgg19(pretrained=True).features
model_conv = models.vgg19(pretrained=True)
# for name, child in model_conv.named_children():
#     for name2, params in child.named_parameters():
#         print(name, name2)


model= copy.deepcopy(model.eval())
model = model[:17]

old_weights=list(model_conv.parameters())[0]
old_bias=list(model_conv.parameters())[1]
# interest_wight=list(old_weights.parameters())[0].sum(dim=1, keepdim=True)
interest_wight=list(model_conv.parameters())[0].sum(dim=1, keepdim=True)
model_conv.features[0]= nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
# list(model_conv.features[0].parameters())[0]=interest_wight
list(model_conv.parameters())[0]=old_weights[:,0:1,:,:]
list(model_conv.parameters())[1]=old_bias
# create custom Alexnet
class CustomModelnet(nn.Module):
    def __init__(self, num_classes):
        super(CustomModelnet, self).__init__()
        self.features = nn.Sequential(*list(model.features.children()))
        self.classifier = nn.Sequential(*[list(model.classifier.children())[i] for i in [1, 2, 4, 5]], nn.Linear(4096, num_classes),
            nn.Sigmoid()
        )
        1

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 256 * 6 * 6)
        x = self.classifier(x)
        return x

# load custom model
model2 = CustomModelnet(num_classes=10)





#
# # load the original Alexnet model
# model = models.alexnet(pretrained=True)
#
# # create custom Alexnet
# class CustomAlexnet(nn.Module):
#     def __init__(self, num_classes):
#         super(CustomAlexnet, self).__init__()
#         self.features = nn.Sequential(*list(model.features.children()))
#         self.classifier = nn.Sequential(*[list(model.classifier.children())[i] for i in [1, 2, 4, 5]], nn.Linear(4096, num_classes),
#             nn.Sigmoid()
#         )
#         1
#
#     def forward(self, x):
#         x = self.features(x)
#         x = x.view(x.size(0), 256 * 6 * 6)
#         x = self.classifier(x)
#         return x
#
# # load custom model
# model2 = CustomAlexnet(num_classes=10)