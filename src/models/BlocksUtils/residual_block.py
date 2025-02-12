""" 
Residual convolutional block
"""

import torch.nn as nn
import torch.nn.functional as F


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)



class ResidualBlock(nn.Module):
    """
    Residual block a-la ResNet
    """
    expansion = 1

    def __init__(self, in_planes, out_planes, downsample_factor=1,
                 batch_norm=True, drop_final_activation=False):
        """ """
        super().__init__()
        self.conv1 = conv3x3(in_planes, out_planes, stride=1)
        self.bn1 = nn.BatchNorm2d(out_planes) if batch_norm else nn.Identity()
        self.relu = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = conv3x3(out_planes, out_planes)
        self.bn2 = nn.BatchNorm2d(out_planes) if batch_norm else nn.Identity()
        self.downsample_factor = downsample_factor
        self.drop_final_activation = drop_final_activation

        self.downsample = None
        if self.downsample_factor != 1 or in_planes != out_planes:
            self.downsample = nn.Sequential(
                conv1x1(in_planes, out_planes, stride=1),
                nn.AvgPool2d(downsample_factor),
                nn.BatchNorm2d(out_planes) if batch_norm else nn.Identity()
            )
        return

    def forward(self, x):
        """ Forward pass """
        identity = x

        out = self.conv1(x)
        out = F.avg_pool2d(out, self.downsample_factor)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity

        if not self.drop_final_activation:
            out = self.relu(out)

        return out
 

