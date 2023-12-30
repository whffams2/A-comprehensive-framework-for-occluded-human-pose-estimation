# ------------------------------------------------------------------------------
# Copyright (c) Microsoft
# Licensed under the MIT License.
# Written by Bin Xiao (Bin.Xiao@microsoft.com)
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import, division
import math
import torch
import torch.nn as nn
import torch.cuda.comm
from utils.attention import ChannelAttention, Spatial_Attention
from utils.non_local_embedded_gaussian import NONLocalBlock2D
from utils.graph_utils import *
from config import cfg
from core.inference import get_gcn_max_preds
from utils.transforms import transform_gcn_preds
import torch.nn.functional as F
import os
import logging

import torch
import torch.nn as nn

BN_MOMENTUM = 0.1
logger = logging.getLogger(__name__)


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes, momentum=BN_MOMENTUM)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion,
                                  momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class HighResolutionModule(nn.Module):
    def __init__(self, num_branches, blocks, num_blocks, num_inchannels,
                 num_channels, fuse_method, multi_scale_output=True):
        super(HighResolutionModule, self).__init__()
        self._check_branches(
            num_branches, blocks, num_blocks, num_inchannels, num_channels)

        self.num_inchannels = num_inchannels
        self.fuse_method = fuse_method
        self.num_branches = num_branches

        self.multi_scale_output = multi_scale_output

        self.branches = self._make_branches(
            num_branches, blocks, num_blocks, num_channels)
        self.fuse_layers = self._make_fuse_layers()
        self.relu = nn.ReLU(True)

    def _check_branches(self, num_branches, blocks, num_blocks,
                        num_inchannels, num_channels):
        if num_branches != len(num_blocks):
            error_msg = 'NUM_BRANCHES({}) <> NUM_BLOCKS({})'.format(
                num_branches, len(num_blocks))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_channels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_CHANNELS({})'.format(
                num_branches, len(num_channels))
            logger.error(error_msg)
            raise ValueError(error_msg)

        if num_branches != len(num_inchannels):
            error_msg = 'NUM_BRANCHES({}) <> NUM_INCHANNELS({})'.format(
                num_branches, len(num_inchannels))
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _make_one_branch(self, branch_index, block, num_blocks, num_channels,
                         stride=1):
        downsample = None
        if stride != 1 or \
                self.num_inchannels[branch_index] != num_channels[branch_index] * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index] * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(
                    num_channels[branch_index] * block.expansion,
                    momentum=BN_MOMENTUM
                ),
            )

        layers = []
        layers.append(
            block(
                self.num_inchannels[branch_index],
                num_channels[branch_index],
                stride,
                downsample
            )
        )
        self.num_inchannels[branch_index] = \
            num_channels[branch_index] * block.expansion
        for i in range(1, num_blocks[branch_index]):
            layers.append(
                block(
                    self.num_inchannels[branch_index],
                    num_channels[branch_index]
                )
            )

        return nn.Sequential(*layers)

    def _make_branches(self, num_branches, block, num_blocks, num_channels):
        branches = []

        for i in range(num_branches):
            branches.append(
                self._make_one_branch(i, block, num_blocks, num_channels)
            )

        return nn.ModuleList(branches)

    def _make_fuse_layers(self):
        if self.num_branches == 1:
            return None

        num_branches = self.num_branches
        num_inchannels = self.num_inchannels
        fuse_layers = []
        for i in range(num_branches if self.multi_scale_output else 1):
            fuse_layer = []
            for j in range(num_branches):
                if j > i:
                    fuse_layer.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_inchannels[j],
                                num_inchannels[i],
                                1, 1, 0, bias=False
                            ),
                            nn.BatchNorm2d(num_inchannels[i]),
                            nn.Upsample(scale_factor=2 ** (j - i), mode='nearest')
                        )
                    )
                elif j == i:
                    fuse_layer.append(None)
                else:
                    conv3x3s = []
                    for k in range(i - j):
                        if k == i - j - 1:
                            num_outchannels_conv3x3 = num_inchannels[i]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3)
                                )
                            )
                        else:
                            num_outchannels_conv3x3 = num_inchannels[j]
                            conv3x3s.append(
                                nn.Sequential(
                                    nn.Conv2d(
                                        num_inchannels[j],
                                        num_outchannels_conv3x3,
                                        3, 2, 1, bias=False
                                    ),
                                    nn.BatchNorm2d(num_outchannels_conv3x3),
                                    nn.ReLU(True)
                                )
                            )
                    fuse_layer.append(nn.Sequential(*conv3x3s))
            fuse_layers.append(nn.ModuleList(fuse_layer))

        return nn.ModuleList(fuse_layers)

    def get_num_inchannels(self):
        return self.num_inchannels

    def forward(self, x):
        if self.num_branches == 1:
            return [self.branches[0](x[0])]

        for i in range(self.num_branches):
            x[i] = self.branches[i](x[i])

        x_fuse = []

        for i in range(len(self.fuse_layers)):
            y = x[0] if i == 0 else self.fuse_layers[i][0](x[0])
            for j in range(1, self.num_branches):
                if i == j:
                    y = y + x[j]
                else:
                    y = y + self.fuse_layers[i][j](x[j])
            x_fuse.append(self.relu(y))

        return x_fuse


class ExtendSkeletonGraphConv(nn.Module):
    """
    G-Motion : Multi-hop Mixed graph convolution layer
    """

    # def __init__(self, in_features, out_features, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, skeleton_graph, bias=True):
    def __init__(self, in_features, out_features, adj, adj_ext1, adj_ext2, adj_ext3, skeleton_graph, bias=True):
        super(ExtendSkeletonGraphConv, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.skeleton_graph = skeleton_graph
        self.NoAffinityModulation = True
        self.W = nn.Parameter(torch.zeros(size=(2, in_features, out_features), dtype=torch.float))
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        if self.skeleton_graph == 1:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 2:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 3:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda4 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
        elif self.skeleton_graph == 4:
            self.Lamda1 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda2 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda3 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda4 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))
            self.Lamda5 = nn.Parameter(torch.ones(size=(adj.size(0), out_features), dtype=torch.float))

        self.adj = adj
        if self.NoAffinityModulation is not True:
            self.adj2 = nn.Parameter(torch.ones_like(adj))
            nn.init.constant_(self.adj2, 1e-6)

        self.adj_ext1 = adj_ext1
        if self.NoAffinityModulation is not True:
            self.adj_ext1_sub = nn.Parameter(torch.ones_like(adj_ext1))
            nn.init.constant_(self.adj_ext1_sub, 1e-6)

        if self.skeleton_graph == 2 or self.skeleton_graph > 2:
            self.adj_ext2 = adj_ext2
            if self.NoAffinityModulation is not True:
                self.adj_ext2_sub = nn.Parameter(torch.ones_like(adj_ext2))
                nn.init.constant_(self.adj_ext2_sub, 1e-6)

        if self.skeleton_graph == 3 or self.skeleton_graph > 3:
            self.adj_ext3 = adj_ext3
            if self.NoAffinityModulation is not True:
                self.adj_ext3_sub = nn.Parameter(torch.ones_like(adj_ext3))
                nn.init.constant_(self.adj_ext3_sub, 1e-6)

        # if self.skeleton_graph == 4:
        #     self.adj_ext4 = adj_ext4
        #     if self.NoAffinityModulation is not True:
        #         self.adj_ext4_sub = nn.Parameter(torch.ones_like(adj_ext4))
        #         nn.init.constant_(self.adj_ext4_sub, 1e-6)

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features, dtype=torch.float))
            stdv = 1. / math.sqrt(self.W.size(2))
            self.bias.data.uniform_(-stdv, stdv)
        else:
            self.register_parameter('bias', None)

    def forward(self, input):
        h0 = torch.matmul(input, self.W[0])
        h1 = torch.matmul(input, self.W[1])

        if self.NoAffinityModulation is not True:
            adj = self.adj.to(input.device) + self.adj2.to(input.device)
        else:
            adj = self.adj.to(input.device)
        adj = (adj.T + adj) / 2

        if self.NoAffinityModulation is not True:
            adj_ext1 = self.adj_ext1.to(input.device) + self.adj_ext1_sub.to(input.device)
        else:
            adj_ext1 = self.adj_ext1.to(input.device)
        adj_ext1 = (adj_ext1.T + adj_ext1) / 2

        if self.skeleton_graph == 2 or self.skeleton_graph > 2:
            if self.NoAffinityModulation is not True:
                adj_ext2 = self.adj_ext2.to(input.device) + self.adj_ext2_sub.to(input.device)
            else:
                adj_ext2 = self.adj_ext2.to(input.device)
            adj_ext2 = (adj_ext2.T + adj_ext2) / 2

        if self.skeleton_graph == 3 or self.skeleton_graph > 3:
            if self.NoAffinityModulation is not True:
                adj_ext3 = self.adj_ext3.to(input.device) + self.adj_ext3_sub.to(input.device)
            else:
                adj_ext3 = self.adj_ext3.to(input.device)
            adj_ext3 = (adj_ext3.T + adj_ext3) / 2

        # if self.skeleton_graph == 4:
        #     if self.NoAffinityModulation is not True:
        #         adj_ext4 = self.adj_ext4.to(input.device) + self.adj_ext4_sub.to(input.device)
        #     else:
        #         adj_ext4 = self.adj_ext4.to(input.device)
        #     adj_ext4 = (adj_ext4.T + adj_ext4) / 2

        E = torch.eye(adj.size(0), dtype=torch.float).to(input.device)

        # MM-GCN
        if self.skeleton_graph == 4:
            # WHA5 = torch.matmul(adj_ext4 * E, h0) + torch.matmul(adj_ext4 * (1 - E), h1)
            # C5 = self.Lamda5 * WHA5
            WHA4 = torch.matmul(adj_ext3 * E, h0) + torch.matmul(adj_ext3 * (1 - E), h1)
            C4 = self.Lamda4 * WHA4
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            # output_out = C1 + (1 - self.Lamda1) * C2 + ((1 - self.Lamda1) * ((1 - self.Lamda2) * C3)) + \
            #              ((1 - self.Lamda1) * ((1 - self.Lamda2) * ((1 - self.Lamda3) * C4))) + \
            #              ((1 - self.Lamda1) * ((1 - self.Lamda2) * ((1 - self.Lamda3) * ((1 - self.Lamda4) * C5))))
        elif self.skeleton_graph == 3:
            WHA4 = torch.matmul(adj_ext3 * E, h0) + torch.matmul(adj_ext3 * (1 - E), h1)
            C4 = self.Lamda4 * WHA4
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1 - self.Lamda1) * C2 + ((1 - self.Lamda1) * ((1 - self.Lamda2) * C3)) + \
                         ((1 - self.Lamda1) * ((1 - self.Lamda2) * ((1 - self.Lamda3) * C4)))
        elif self.skeleton_graph == 2:
            WHA3 = torch.matmul(adj_ext2 * E, h0) + torch.matmul(adj_ext2 * (1 - E), h1)
            C3 = self.Lamda3 * WHA3
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1 - self.Lamda1) * C2 + ((1 - self.Lamda1) * ((1 - self.Lamda2) * C3))
        elif self.skeleton_graph == 1:
            WHA2 = torch.matmul(adj_ext1 * E, h0) + torch.matmul(adj_ext1 * (1 - E), h1)
            C2 = self.Lamda2 * WHA2
            WHA1 = torch.matmul(adj * E, h0) + torch.matmul(adj * (1 - E), h1)
            C1 = self.Lamda1 * WHA1
            output_out = C1 + (1 - self.Lamda1) * C2

        if self.bias is not None:
            return output_out + self.bias.view(1, 1, -1)
        else:
            return output_out

    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.in_features) + ' -> ' + str(self.out_features) + ')'


class _GraphConv(nn.Module):
    # def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, input_dim, output_dim, p_dropout=None):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, input_dim, output_dim, p_dropout=None, skeleton_graph=1, ):
        super(_GraphConv, self).__init__()

        # self.gconv = ExtendSkeletonGraphConv(input_dim, output_dim, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4)
        self.gconv = ExtendSkeletonGraphConv(input_dim, output_dim, adj, adj_ext1, adj_ext2, adj_ext3,
                                             skeleton_graph=skeleton_graph)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

        if p_dropout is not None:
            self.dropout = nn.Dropout(p_dropout)
        else:
            self.dropout = None

    def forward(self, x):
        x = self.gconv(x).transpose(1, 2)
        x = self.bn(x).transpose(1, 2)
        if self.dropout is not None:
            x = self.dropout(self.relu(x))

        x = self.relu(x)
        return x


class _ResGraphConv(nn.Module):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, input_dim, output_dim, hid_dim, p_dropout):
        super(_ResGraphConv, self).__init__()

        self.gconv1 = _GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, input_dim, hid_dim, p_dropout)
        self.gconv2 = _GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, hid_dim, output_dim, p_dropout)
        self.bn = nn.BatchNorm1d(output_dim)
        self.relu = nn.ReLU()

    def forward(self, x, joints_features=None):

        if joints_features is None:
            residual = x
        else:
            joints_features = joints_features.transpose(1, 2).contiguous()
            x = torch.cat([joints_features, x], dim=2)
            residual = x
        out = self.gconv1(x)
        out = self.gconv2(out)
        out = self.bn(residual.transpose(1, 2).contiguous() + out.transpose(1, 2).contiguous())
        out = self.relu(out)

        return out.transpose(1, 2).contiguous()


blocks_dict = {
    'BASIC': BasicBlock,
    'BOTTLENECK': Bottleneck
}


class MMGCN_ALL(nn.Module):
    # def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, adj_ext4, hid_dim, non_local=False, coords_dim=(2, 3),
    #              num_layers=4, skeleton_graph=1, p_dropout=None):
    def __init__(self, adj, adj_ext1, adj_ext2, adj_ext3, skeleton_graph, hid_dim, non_local=False, coords_dim=(2, 2),
                 num_layers=4, p_dropout=None):
        super(MMGCN_ALL, self).__init__()

        self.non_local = non_local
        _gconv_input = [
            _GraphConv(adj, adj_ext1, adj_ext2, adj_ext3, coords_dim[0], hid_dim, skeleton_graph=skeleton_graph,
                       p_dropout=p_dropout)]
        _gconv_layers = []
        self.level_conv = nn.Conv2d(17, 17, 3, stride=1, padding=1)
        # self.level_conv = nn.Conv2d(14, 14, 3, stride=1, padding=1)
        self.FC = nn.Sequential(nn.Sequential(nn.Linear(128 * 96, 1024), nn.ReLU(inplace=True)),
                                nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(inplace=True)),
                                nn.Linear(1024, 2))

        for i in range(num_layers - 1):
            _gconv_layers.append(_ResGraphConv(adj, adj_ext1, adj_ext2, adj_ext3, hid_dim, hid_dim, hid_dim,
                                               p_dropout=p_dropout))

        self.gconv_layer_last = _ResGraphConv(adj, adj_ext1, adj_ext2, adj_ext3, hid_dim + 17, hid_dim + 17,
                                              hid_dim + 17,
                                              p_dropout=p_dropout)
        # self.gconv_layer_last = _ResGraphConv(adj, adj_ext1, adj_ext2, adj_ext3, hid_dim + 14, hid_dim + 14,
        #                                       hid_dim + 14,
        #                                       p_dropout=p_dropout)
        self.gconv_input = nn.Sequential(*_gconv_input)
        self.gconv_layers = nn.Sequential(*_gconv_layers)

        self.gconv_output = ExtendSkeletonGraphConv(hid_dim + 17, coords_dim[1], adj, adj_ext1, adj_ext2, adj_ext3,
                                                    skeleton_graph=skeleton_graph)
        # self.gconv_output = ExtendSkeletonGraphConv(hid_dim + 14, coords_dim[1], adj, adj_ext1, adj_ext2, adj_ext3,
        #                                             skeleton_graph=skeleton_graph)
        if self.non_local:
            self.non_local = NONLocalBlock2D(in_channels=hid_dim + 17, sub_sample=False)
            # self.non_local = NONLocalBlock2D(in_channels=hid_dim + 14, sub_sample=False)

    def forward(self, x, hm, features):
        batch_size = x.shape[0]
        if self.non_local is False:
            out = self.gconv_input(x)
            out = self.gconv_layers(out)
            out = self.gconv_output(out)
        else:
            # x = 32 17 2  hm 32 17 2   features 32 17 64 48
            heatmap = self.level_conv(features)  # 32 17 64 48
            bs = heatmap.shape[0]
            heat_map_intergral = self.FC(heatmap.view(bs * 17, -1)).view(bs, 34)
            # heat_map_intergral = self.FC(heatmap.view(bs * 14, -1)).view(bs, 28)
            hm_4 = heat_map_intergral.view(-1, 17, 2)
            # hm_4 = heat_map_intergral.view(-1, 14, 2)
            j_1_4 = F.grid_sample(heatmap, hm_4[:, None, :, :]).squeeze(2)  # 32 17 17
            out = self.gconv_input(x)  # 32 17 128
            out = self.gconv_layers(out)  # 32 17 128
            out = self.gconv_layer_last(out, joints_features=j_1_4)
            out = out.unsqueeze(2)  # 32 17 1 128
            out = out.permute(0, 3, 2, 1)  # 32 128 1 17
            out = self.non_local(out)  # 32 128 1 17
            out = out.permute(0, 3, 1, 2)  # 32 17 128 1
            out = out.squeeze()  # 32 17 128
            out = self.gconv_output(out)  # 32 17 2
        return out


class PoseHighResolutionNetALL(nn.Module):

    def __init__(self, cfg, **kwargs):
        self.inplanes = 64
        extra = cfg['MODEL']['EXTRA']
        super(PoseHighResolutionNetALL, self).__init__()

        # stem net
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(64, momentum=BN_MOMENTUM)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(Bottleneck, 64, 4)
        self.scale_factor = cfg.MODEL.UP_SCALE
        self.stage2_cfg = extra['STAGE2']
        num_channels = self.stage2_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage2_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition1 = self._make_transition_layer([256], num_channels)
        self.stage2, pre_stage_channels = self._make_stage(
            self.stage2_cfg, num_channels)

        self.stage3_cfg = extra['STAGE3']
        num_channels = self.stage3_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage3_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition2 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage3, pre_stage_channels = self._make_stage(
            self.stage3_cfg, num_channels)

        self.stage4_cfg = extra['STAGE4']
        num_channels = self.stage4_cfg['NUM_CHANNELS']
        block = blocks_dict[self.stage4_cfg['BLOCK']]
        num_channels = [
            num_channels[i] * block.expansion for i in range(len(num_channels))
        ]
        self.transition3 = self._make_transition_layer(
            pre_stage_channels, num_channels)
        self.stage4, pre_stage_channels = self._make_stage(
            self.stage4_cfg, num_channels, multi_scale_output=False)
        self.multi_conv = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        self.kpt_conv = nn.Conv2d(
            in_channels=pre_stage_channels[0],
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        self.channel_att = ChannelAttention(17, 17)
        # self.spatial_att = Spatial_Attention(32)
        self.enhance_feature = nn.Conv2d(
            in_channels=17 * 2,
            out_channels=cfg['MODEL']['NUM_JOINTS'],
            kernel_size=extra['FINAL_CONV_KERNEL'],
            stride=1,
            padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        )
        # self.final_layer = nn.Conv2d(
        #     in_channels=pre_stage_channels[0],
        #     out_channels=cfg['MODEL']['NUM_JOINTS'],
        #     kernel_size=extra['FINAL_CONV_KERNEL'],
        #     stride=1,
        #     padding=1 if extra['FINAL_CONV_KERNEL'] == 3 else 0
        # )

        self.pretrained_layers = extra['PRETRAINED_LAYERS']

    def _make_transition_layer(
            self, num_channels_pre_layer, num_channels_cur_layer):
        num_branches_cur = len(num_channels_cur_layer)
        num_branches_pre = len(num_channels_pre_layer)

        transition_layers = []
        for i in range(num_branches_cur):
            if i < num_branches_pre:
                if num_channels_cur_layer[i] != num_channels_pre_layer[i]:
                    transition_layers.append(
                        nn.Sequential(
                            nn.Conv2d(
                                num_channels_pre_layer[i],
                                num_channels_cur_layer[i],
                                3, 1, 1, bias=False
                            ),
                            nn.BatchNorm2d(num_channels_cur_layer[i]),
                            nn.ReLU(inplace=True)
                        )
                    )
                else:
                    transition_layers.append(None)
            else:
                conv3x3s = []
                for j in range(i + 1 - num_branches_pre):
                    inchannels = num_channels_pre_layer[-1]
                    outchannels = num_channels_cur_layer[i] \
                        if j == i - num_branches_pre else inchannels
                    conv3x3s.append(
                        nn.Sequential(
                            nn.Conv2d(
                                inchannels, outchannels, 3, 2, 1, bias=False
                            ),
                            nn.BatchNorm2d(outchannels),
                            nn.ReLU(inplace=True)
                        )
                    )
                transition_layers.append(nn.Sequential(*conv3x3s))

        return nn.ModuleList(transition_layers)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(
                    self.inplanes, planes * block.expansion,
                    kernel_size=1, stride=stride, bias=False
                ),
                nn.BatchNorm2d(planes * block.expansion, momentum=BN_MOMENTUM),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def _make_stage(self, layer_config, num_inchannels,
                    multi_scale_output=True):
        num_modules = layer_config['NUM_MODULES']
        num_branches = layer_config['NUM_BRANCHES']
        num_blocks = layer_config['NUM_BLOCKS']
        num_channels = layer_config['NUM_CHANNELS']
        block = blocks_dict[layer_config['BLOCK']]
        fuse_method = layer_config['FUSE_METHOD']

        modules = []
        for i in range(num_modules):
            # multi_scale_output is only used last module
            if not multi_scale_output and i == num_modules - 1:
                reset_multi_scale_output = False
            else:
                reset_multi_scale_output = True

            modules.append(
                HighResolutionModule(
                    num_branches,
                    block,
                    num_blocks,
                    num_inchannels,
                    num_channels,
                    fuse_method,
                    reset_multi_scale_output
                )
            )
            num_inchannels = modules[-1].get_num_inchannels()

        return nn.Sequential(*modules), num_inchannels

    def _sample_feats(self, features, pos_ind):
        feats = features[:, :, pos_ind[0] // 2:pos_ind[0] // 2 + 1, pos_ind[1] // 2:pos_ind[1] // 2 + 1]
        # B, C, H, W = features.shape
        # feats = []
        # for i in range(B):
        #     feat = features[i, :, int(pos_ind[i, 0]), int(pos_ind[i, 1])]
        #     feats.append(feat)
        # feats = torch.stack(feats)
        return feats

    ''' 
    # def generate_2d_integral_preds_tensor(self, heatmaps, num_joints, x_dim, y_dim):
    #     assert isinstance(heatmaps, torch.Tensor)
    #
    #     heatmaps = heatmaps.reshape((heatmaps.shape[0], num_joints, y_dim, x_dim))
    #
    #     accu_x = heatmaps.sum(dim=2)
    #     accu_y = heatmaps.sum(dim=3)
    #
    #     accu_x = accu_x * torch.cuda.comm.broadcast(torch.arange(x_dim).type(torch.cuda.FloatTensor),
    #                                                 devices=[accu_x.device.index])[0]
    #     accu_y = accu_y * torch.cuda.comm.broadcast(torch.arange(y_dim).type(torch.cuda.FloatTensor),
    #                                                 devices=[accu_y.device.index])[0]
    #
    #     accu_x = accu_x.sum(dim=2, keepdim=True)
    #     accu_y = accu_y.sum(dim=2, keepdim=True)
    #
    #     return accu_x, accu_y

    # def softmax_integral_tensor(self, preds, num_joints, hm_width, hm_height):
    #     # global soft max
    #     preds = preds.reshape((preds.shape[0], num_joints, -1))
    #     preds = F.softmax(preds, 2)
    #     score = torch.max(preds, -1)[0]
    #
    #     # integrate heatmap into joint location
    #
    #     x, y = self.generate_2d_integral_preds_tensor(preds, num_joints, hm_width, hm_height)
    #     x = x / float(hm_width) - 0.5
    #     y = y / float(hm_height) - 0.5
    #
    #     preds = torch.cat((x, y), dim=2)
    #     preds *= 2
    #     preds = preds.reshape((preds.shape[0], num_joints * 2))
    #     return preds, score
'''

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.layer1(x)

        x_list = []
        for i in range(self.stage2_cfg['NUM_BRANCHES']):
            if self.transition1[i] is not None:
                x_list.append(self.transition1[i](x))
            else:
                x_list.append(x)
        y_list = self.stage2(x_list)

        x_list = []
        for i in range(self.stage3_cfg['NUM_BRANCHES']):
            if self.transition2[i] is not None:
                x_list.append(self.transition2[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage3(x_list)

        x_list = []
        for i in range(self.stage4_cfg['NUM_BRANCHES']):
            if self.transition3[i] is not None:
                x_list.append(self.transition3[i](y_list[-1]))
            else:
                x_list.append(y_list[i])
        y_list = self.stage4(x_list)
        init_feature = y_list[0]
        # multi_peek features
        multi_heatmap = self.multi_conv(init_feature)
        # channel attention
        instance_coord = np.array((multi_heatmap.shape[2], multi_heatmap.shape[3]))
        instance_param = self._sample_feats(multi_heatmap, instance_coord)  # 1 17 1 1
        B, C, H, W = instance_param.shape
        instance_param = instance_param.reshape(B * C, -1)
        linear = nn.Linear(instance_param.shape[1], 1, bias=False).to(instance_param.device)
        instance_param = linear(instance_param).reshape(B, C)
        enhance_features = self.channel_att(multi_heatmap, instance_param)
        # spatial attention
        # spatial_att_features = self.spatial_att(init_feature)
        kpt_features = self.kpt_conv(init_feature)
        final_features = torch.cat([kpt_features, enhance_features], dim=1)
        out = self.enhance_feature(final_features)
        if self.scale_factor > 1:
            out = F.interpolate(out, scale_factor=self.scale_factor, mode="bilinear",
                                align_corners=False)
            multi_heatmap = F.interpolate(multi_heatmap, scale_factor=self.scale_factor, mode="bilinear",
                                          align_corners=False)

        return out, multi_heatmap

    def init_weights(self, pretrained=''):
        logger.info('=> init weights from normal distribution')
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.ConvTranspose2d):
                nn.init.normal_(m.weight, std=0.001)
                for name, _ in m.named_parameters():
                    if name in ['bias']:
                        nn.init.constant_(m.bias, 0)

        if os.path.isfile(pretrained):
            pretrained_state_dict = torch.load(pretrained)
            logger.info('=> loading pretrained model {}'.format(pretrained))

            need_init_state_dict = {}
            for name, m in pretrained_state_dict.items():
                if name.split('.')[0] in self.pretrained_layers \
                        or self.pretrained_layers[0] is '*':
                    need_init_state_dict[name] = m
            self.load_state_dict(need_init_state_dict, strict=False)
        elif pretrained:
            logger.error('=> please download pre-trained models first!')
            raise ValueError('{} is not exist!'.format(pretrained))


def get_pose_net(cfg, is_train, **kwargs):
    model = PoseHighResolutionNetALL(cfg, **kwargs)

    if is_train and cfg['MODEL']['INIT_WEIGHTS']:
        model.init_weights(cfg['MODEL']['PRETRAINED'])

    return model


def get_gcn_net(cfg, adj, adj_ext_1, adj_ext_2, adj_ext_3, is_train, **kwargs):
    model = MMGCN_ALL(adj, adj_ext_1, adj_ext_2, adj_ext_3, skeleton_graph=cfg['GCN']['SKELETON_GRAPH'],
                      hid_dim=cfg['GCN']['CHANNELS'], non_local=cfg['GCN']['Non_Local'],
                      num_layers=cfg['GCN']['NUM_LAYERS'], p_dropout=cfg['GCN']['P_DROPOUT'])

    return model
