
from utilities.simEps_unet import UNet, UNet_f, UNet_g, UNet_h


def net_factory(net_type="unet_f", in_chns=3, class_num=6):
    if net_type == "unet_f":
        net = UNet_f(in_chns=in_chns, class_num=class_num).cuda()
    
    elif net_type == "unet_g":
        net = UNet_g(in_chns=in_chns, class_num=class_num).cuda()
    
    elif net_type == "unet_h":
        net = UNet_h(in_chns=in_chns, class_num=class_num).cuda()
    else:
        net = None
    return net
