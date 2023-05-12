#The model for dual-segmetnation netwrok
#Import the required libraries
import os
# from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import functools
from utilities.net_factory import net_factory
 
def create_model(ema=False):
    # Network definition
    model = net_factory(net_type='unet', in_chns=1, class_num=7)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()
# ema_model = create_model(ema=True)
# model1 = DeepLabV3Plus("mobilenet_v2", encoder_weights= None, classes=7, in_channels = 1, activation=None)