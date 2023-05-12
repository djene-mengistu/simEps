
#Import the required libraries
import os
import torch
import torch.nn as nn
import functools
from utilities.net_factory import net_factory

# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used

def create_model(ema=False):
    # Network definition
    model = net_factory(net_type='unet', in_chns=3, class_num=6)
    
    if ema:
        for param in model.parameters():
            param.detach_()
    return model

model = create_model()
# ema_model = create_model(ema=True)