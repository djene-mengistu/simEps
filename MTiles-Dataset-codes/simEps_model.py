
#Import the required libraries
import os
# from segmentation_models_pytorch.deeplabv3 import model
import torch
import torch.nn as nn
import functools
from utilities.simEps_net_factory import net_factory


model1 = net_factory(net_type='unet_f', in_chns=3, class_num=6) 
model2 = net_factory(net_type='unet_f', in_chns=3, class_num=6) 
model3 = net_factory(net_type='unet_f', in_chns=3, class_num=6) 
# model2 = xavier_normal_init_weight(model2)
# model3 = xavier_uniform_init_weight(model3)
# model3 = net_factory(net_type='unet_h', in_chns=3, class_num=4)

model1 = nn.DataParallel(model1, device_ids=[0,1])
model2 = nn.DataParallel(model2, device_ids=[0,1])
model3 = nn.DataParallel(model3, device_ids=[0,1])
# model1 = nn.DataParallel(model1)
# model2 = nn.DataParallel(model2)
# model3 = nn.DataParallel(model3)
# print(model)