#The model for dual-segmetnation netwrok
#Import the required libraries
# import os
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # specify which GPU(s) to be used
# from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import functools
from utilities.simEps_net_factory import net_factory 
 

model1 = net_factory(net_type='unet_f', in_chns=1, class_num=7) #normal unet
model2 = net_factory(net_type='unet_f', in_chns=1, class_num=7) #Random feature droput unet with xavier normal init
model3 = net_factory(net_type='unet_f', in_chns=1, class_num=7) #Random noise UNet with Xavier uniform init
# model2 = xavier_normal_init_weight(model2)
# model3 = xavier_uniform_init_weight(model3)
# model3 = net_factory(net_type='unet_h', in_chns=3, class_num=4)

model1 = nn.DataParallel(model1, device_ids=[0,1])
model2 = nn.DataParallel(model2, device_ids=[0,1])
model3 = nn.DataParallel(model3, device_ids=[0,1])
