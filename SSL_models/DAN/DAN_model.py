#The model for dual-segmetnation netwrok
#Import the required libraries
import os
from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from net_factory import net_factory

model = net_factory(net_type='unet', in_chns=3, class_num=4)


class FCDiscriminator(nn.Module):

    def __init__(self, num_classes, ndf=64, n_channel=3):
        super(FCDiscriminator, self).__init__()
        self.conv0 = nn.Conv2d(
            num_classes, ndf, kernel_size=4, stride=2, padding=1)
        self.conv1 = nn.Conv2d(
            n_channel, ndf, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(ndf, ndf*2, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(
            ndf*2, ndf*4, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(
            ndf*4, ndf*8, kernel_size=4, stride=2, padding=1)
        self.classifier = nn.Linear(ndf*32, 2)
        self.avgpool = nn.AvgPool2d((7, 7))

        self.leaky_relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.dropout = nn.Dropout2d(0.5)
        # self.up_sample = nn.Upsample(scale_factor=32, mode='bilinear')
        # self.sigmoid = nn.Sigmoid()

    def forward(self, map, feature):
        map_feature = self.conv0(map)
        image_feature = self.conv1(feature)
        x = torch.add(map_feature, image_feature)

        x = self.conv2(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv3(x)
        x = self.leaky_relu(x)
        x = self.dropout(x)

        x = self.conv4(x)
        x = self.leaky_relu(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = self.up_sample(x)
        # x = self.sigmoid(x)

        return x

DAN = FCDiscriminator(num_classes = 4)
# x = torch.randn(16,4,256,256)
# y = torch.randn(16,3,256,256)

# DAN_out = DAN(x, y)
# print(DAN_out.shape)
