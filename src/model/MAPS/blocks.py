"""Convolution blocks used by the ESCAP backbone."""

import torch
from torch import nn
from torch.nn import functional as F


IN_PLACE = True


class StdConv3d(nn.Conv3d):
    def forward(self, x):
        weight = self.weight
        weight_mean = weight.mean(dim=1, keepdim=True).mean(dim=2, keepdim=True).mean(dim=3, keepdim=True).mean(dim=4, keepdim=True)
        weight = weight - weight_mean
        std = torch.sqrt(torch.var(weight.view(weight.size(0), -1), dim=1) + 1e-12).view(-1, 1, 1, 1, 1)
        weight = weight / std.expand_as(weight)
        return F.conv3d(x, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


def conv3x3x3(in_planes, out_planes, kernel_size=(3, 3, 3), stride=(1, 1, 1), padding=1, dilation=1,
              bias=False, weight_std=False):
    if weight_std:
        return StdConv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                         padding=padding, dilation=dilation, bias=bias)
    return nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                     padding=padding, dilation=dilation, bias=bias)


class NoBottleneck(nn.Module):
    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, multi_grid=1, weight_std=False):
        super().__init__()
        self.gn1 = nn.GroupNorm(16, inplanes)
        self.conv1 = conv3x3x3(inplanes, planes, kernel_size=(3, 3, 3), stride=stride,
                               padding=(1, 1, 1), dilation=dilation * multi_grid,
                               bias=False, weight_std=weight_std)
        self.relu = nn.ReLU(inplace=IN_PLACE)
        self.gn2 = nn.GroupNorm(16, planes)
        self.conv2 = conv3x3x3(planes, planes, kernel_size=(3, 3, 3), stride=1,
                               padding=(1, 1, 1), dilation=dilation * multi_grid,
                               bias=False, weight_std=weight_std)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(self.relu(self.gn1(x)))
        out = self.conv2(self.relu(self.gn2(out)))
        if self.downsample is not None:
            residual = self.downsample(x)
        return out + residual
