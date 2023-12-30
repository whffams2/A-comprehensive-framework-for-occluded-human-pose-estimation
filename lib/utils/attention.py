import cv2
import torch
import torch.nn as nn
import numpy as np
import torch
import math
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveMaxPool2d, AdaptiveAvgPool2d, \
    NLLLoss, BCELoss, CrossEntropyLoss, AvgPool2d, MaxPool2d, Parameter, Linear, Sigmoid, Softmax, Dropout, Embedding
from torch.nn import functional as F
from torch.autograd import Variable


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ChannelAttention, self).__init__()
        self.atn = nn.Linear(in_channels, out_channels)

    def forward(self, global_features, instance_params):
        B, C, H, W = global_features.size()
        instance_params = self.atn(instance_params).reshape(B, C, 1, 1)  # torch.Size([13, 32, 1, 1])

        # attention
        return global_features * instance_params.expand_as(global_features)


class Spatial_Attention(Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, downsample=False):
        super(Spatial_Attention, self).__init__()
        self.chanel_in = in_dim
        self.downsample = downsample
        self.query_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = Parameter(torch.zeros(1))

        self.softmax = Softmax(dim=-1)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        if self.downsample:
            x =  F.interpolate(x, scale_factor=0.5, mode="bilinear",
                                         align_corners=False)
        m_batchsize, C, height, width = x.shape
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)
        out = (self.gamma * out) * 0.01 + x
        if self.downsample:
            out = F.interpolate(out, scale_factor=2, mode="bilinear",
                                         align_corners=False)
        return out
