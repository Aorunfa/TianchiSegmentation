import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .backbone import ConvBnReLU, UHRNet_W18_Small, UHRNet_W48
BN_MOMENTUM = 0.1

class UHRnet(nn.Module):
    def __init__(self, num_classes=21, backbone='UHRNet_W18_Small'):
        super(UHRnet, self).__init__()
        if backbone == 'UHRNet_W18_Small':
            self.backbone       = UHRNet_W18_Small()
            last_inp_channels   = int(279)

        if backbone == 'UHRNet_W48':
            self.backbone       = UHRNet_W48()
            last_inp_channels   = int(744)

        self.head = nn.Sequential()
        self.head.add_module("conv_1",
            ConvBnReLU(in_channels=last_inp_channels, out_channels=last_inp_channels, kernel_size=1, stride=1, padding=0, bias=True),
        )
        self.head.add_module("cls",
            nn.Conv2d(in_channels=last_inp_channels, out_channels=num_classes, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, inputs):
        H, W = inputs.size(2), inputs.size(3)
        x = self.backbone(inputs)

        x = self.head(x)
        x = F.interpolate(x, size=(H, W), mode='bilinear', align_corners=True)
        return x

def init_weights(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)

if __name__ == '__main__':
    # 测试
    model = UHRnet(num_classes=1)
    init_weights(model)
    imgs = torch.zeros([6, 3, 512, 512])  # b c w h
    outputs = model(imgs)
    print(outputs)
    print(outputs.shape)
    # 权重初始化
