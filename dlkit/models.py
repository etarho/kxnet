# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.6.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# # HPSegNet

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from dlkit.blocks import ResBlock, CBR, GlobalAvgPool2d, AttentionBlock


def _build_resblock(channel_out, channel_in=None):
    if channel_in is None:
        channel_in = channel_out
    return ResBlock(channel_in, channel_out)


# ## Define Network Architecture

class ResNet50(nn.Module):

    def __init__(self):
        super().__init__()
        self.resnet = torchvision.models.resnet50(pretrained=True)
        self.fc = nn.Linear(1000, 100)
        self.out = nn.Linear(100, 9)

    def forward(self, x):
        h = self.resnet(x)
        h = F.relu(h)
        h = self.fc(h)
        h = F.relu(h)
        h = self.out(h)
        y = torch.sigmoid(h)
        return y


class ResNet50_448(nn.Module):

    def __init__(self):
        super().__init__()
        self.cbr1 = CBR(3, 64, kernel_size=7, stride=2, padding=3)
        self.cbr2 = CBR(64, 64, kernel_size=7, stride=2, padding=3)
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.block0 = _build_resblock(256, channel_in=64)
        self.block1 = nn.ModuleList(
            [_build_resblock(256) for _ in range(2)])

        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=2)
        self.block2 = nn.ModuleList(
            [_build_resblock(512) for _ in range(4)])

        self.conv3 = nn.Conv2d(512, 1024, kernel_size=1, stride=2)
        self.block3 = nn.ModuleList(
            [_build_resblock(1024) for _ in range(6)])

        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=1, stride=2)
        self.block4 = nn.ModuleList(
            [_build_resblock(2048) for _ in range(3)])

        self.avg_pool = GlobalAvgPool2d()
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.out = nn.Linear(100, 9)

    def forward(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h = self.pool(h)
        h = self.block0(h)
        for block in self.block1:
            h = block(h)
        h = self.conv2(h)
        for block in self.block2:
            h = block(h)
        h = self.conv3(h)
        for block in self.block3:
            h = block(h)
        h = self.conv4(h)
        for block in self.block4:
            h = block(h)
        h = self.avg_pool(h)
        h = self.fc1(h)
        h = F.relu(h)
        h = self.fc2(h)
        h = F.relu(h)
        h = self.out(h)
        y = torch.sigmoid(h)
        return y


class AttentionResNet50_448(nn.Module):

    def __init__(self):
        super().__init__()
        self.res = ResNet50_448()
        self.atmap1 = AttentionBlock(64)

    def forward(self, x):
        h = self.res.cbr1(x)
        h = self.atmap1(h)
        h = self.res.cbr2(h)
        h = self.res.pool(h)
        h = self.res.block0(h)
        for block in self.res.block1:
            h = block(h)
        h = self.res.conv2(h)
        for block in self.res.block2:
            h = block(h)
        h = self.res.conv3(h)
        for block in self.res.block3:
            h = block(h)
        h = self.res.conv4(h)
        for block in self.res.block4:
            h = block(h)
        h = self.res.avg_pool(h)
        h = self.res.fc1(h)
        h = F.relu(h)
        h = self.res.fc2(h)
        h = F.relu(h)
        h = self.res.out(h)
        y = torch.sigmoid(h)

        return y
