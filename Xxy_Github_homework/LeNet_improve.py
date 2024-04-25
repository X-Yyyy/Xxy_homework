#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/11 8:41
# @Author  : Xxy.
# @FileName: LeNet_improve.py
# @Software: PyCharm
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
class OctConv(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0,padding3 = 3,padding0=0, alpha_in=0.25, alpha_out=0.5):
        super(OctConv, self).__init__()
        self.alpha_in = alpha_in
        self.alpha_out = alpha_out
        self.conv_hh = nn.Conv2d(math.floor(in_channels * (1 - alpha_in)), math.floor(out_channels * (1 - alpha_out)), kernel_size, stride, padding)
        self.conv_hl = nn.Conv2d(math.floor(in_channels * (1 - alpha_in)), math.ceil(out_channels * alpha_out), kernel_size, stride, padding=padding3)
        self.conv_lh = nn.Conv2d(math.ceil(in_channels * alpha_in), math.floor(out_channels * (1 - alpha_out)), kernel_size, stride, padding=padding0)
        self.conv_ll = nn.Conv2d(math.ceil(in_channels * alpha_in), math.ceil(out_channels * alpha_out), kernel_size, stride, padding)

    def forward(self, x):
        x_h, x_l = x
        x_hh = self.conv_hh(x_h)
        x_ll = self.conv_ll(x_l)
        x_hl = F.interpolate(self.conv_hl(x_h), scale_factor=2, mode='nearest')
        x_lh = F.avg_pool2d(self.conv_lh(x_l), 2)
        # print(x_hh.shape)
        # print(x_hl.shape)
        # print(x_lh.shape)
        # print(x_ll.shape)
        if x_hl.shape[-1] != x_ll.shape[-1]:
            x_hl = F.pad(x_hl, [0, x_ll.shape[-1] - x_hl.shape[-1], 0, 0])
        return x_hh + x_lh, x_ll + x_hl
class MultiRF2_Net(nn.Module):
    """
    Loading model parameters: 0.420 Mb
    Multi_resolution_feature_fusion
    """
    def __init__(self, input_channels, input_size, n_classes):
        super(MultiRF2_Net, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 18, 5, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.oct_conv0 = OctConv(input_channels + 6, input_channels + 6, 5, padding=2,
                                 alpha_in=input_channels / (input_channels + 6),
                                 alpha_out=input_channels / (input_channels + 6),padding3 = 2,padding0=2)
        self.oct_conv1 = OctConv(6+16, 6+16, 5, padding=2, alpha_in=0.25, alpha_out=0.25)
        self.fc1 = nn.Linear(16*int((input_size*7/32)*(input_size*7/32)), 120) # input varia a seconda della dimensione dell'immagine di input passare l'input size
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        :param x: [B,N,W,H]
        :return:
        """
        # x = torch.unsqueeze(x,dim=1)
        # print(x.shape) # torch.Size([128, 40, 64, 64])
        x1= F.relu(self.conv1(x))
        # print(x1.shape)
        x_l = F.max_pool2d(x1, (2, 2)) #da 64x64 -> 32x32
        # print(x.shape) #torch.Size([128, 6, 32, 32])
        ###
        x_l_h0, x_l0 = self.oct_conv0((x_l, x))
        x_l0_h = F.max_pool2d(F.relu(self.conv1(x_l0)), (2, 2))
        x_l = x_l_h0
        ###
        x_h = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x_l))), (2, 2)) # 32x32 -> 14x14 -> linear
        # print(x.shape) #torch.Size([128, 16, 14, 14])
        ####
        x_hh_lh, x_ll_hl = self.oct_conv1((x_h,x_l))
        x_ll_hl_h = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x_ll_hl))), (2, 2)) # 32x32 -> 14x14 -> linear
        # x = x_hh_lh + x_ll_hl_h
        x = x_hh_lh
        # x = x_h ## skip connection
        ####
        flat_feature_x = self.num_flat_features(x)
        # print(flat_feature_x) #tensor(3136)
        x = x.view(-1, flat_feature_x)
        # print(x.shape) #torch.Size([128, 3136])
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape) #torch.Size([128, 12])
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))

class MultiRF2_Net_Leap(nn.Module):
    """
    Loading model parameters: 0.420 Mb
    Multi_resolution_feature_fusion
    """
    def __init__(self, input_channels, input_size, n_classes):
        super(MultiRF2_Net_Leap, self).__init__()

        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, 5, padding=0)
        self.conv2_drop = nn.Dropout2d()
        self.oct_conv0 = OctConv(input_channels + 6, input_channels + 6, 5, padding=2,
                                 alpha_in=input_channels / (input_channels + 6),
                                 alpha_out=input_channels / (input_channels + 6),padding3 = 2,padding0=2)
        self.oct_conv1 = OctConv(6+16, 6+16, 5, padding=2, alpha_in=0.25, alpha_out=0.25)
        # self.fc1 = nn.Linear(16*int((input_size*7/32)*(input_size*7/32)), 120) # input varia a seconda della dimensione dell'immagine di input passare l'input size
        self.fc1 = nn.Linear(4352, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, n_classes)

    def forward(self, x):
        """
        :param x: [B,N,W,H]
        :return:
        """
        #
        x = torch.unsqueeze(x,dim=1)
        # print(x.shape) # torch.Size([128, 40, 64, 64])
        x1= F.relu(self.conv1(x))
        # print(x1.shape)
        x_l = F.max_pool2d(x1, (2, 2)) #da 64x64 -> 32x32
        # print(x.shape) #torch.Size([128, 6, 32, 32])
        ###
        x_l_h0, x_l0 = self.oct_conv0((x_l, x))
        x_l0_h = F.max_pool2d(F.relu(self.conv1(x_l0)), (2, 2))
        x_l = x_l_h0
        ###
        x_h = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x_l))), (2, 2)) # 32x32 -> 14x14 -> linear
        # print(x.shape) #torch.Size([128, 16, 14, 14])
        ####
        x_hh_lh, x_ll_hl = self.oct_conv1((x_h,x_l))
        x_ll_hl_h = F.max_pool2d(F.relu(self.conv2_drop(self.conv2(x_ll_hl))), (2, 2)) # 32x32 -> 14x14 -> linear
        # x = x_hh_lh + x_ll_hl_h
        x = x_hh_lh
        # x = x_h ## skip connection
        ####
        flat_feature_x = self.num_flat_features(x)
        # print(flat_feature_x) #tensor(3136)
        x = x.view(-1, flat_feature_x)
        # print(x.shape) #torch.Size([128, 3136])
        x = F.dropout(F.relu(self.fc1(x)), training=self.training)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        # print(x.shape) #torch.Size([128, 12])
        return x

    def num_flat_features(self, x):
        return torch.prod(torch.tensor(x.size()[1:]))
if __name__ == '__main__':
    # LeNet_Improve()
    input = torch.randn([2, 120, 64, 64])
    model = MultiRF2_Net(120, 64, 12)
    # model = LeNet(120,64,12)
    out = model(input)
    print(out.shape)