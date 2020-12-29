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
from pytorch_memlab import profile


# ## Define Network Architecture

class HPNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2), padding=1)

        self.block0 = self._building_block(256, channel_in=64)
        self.block1 = nn.ModuleList([self._building_block(256) for _ in range(2)])

        self.conv2 = nn.Conv2d(256, 512, kernel_size=(1, 1), stride=(2, 2))
        self.block2 = nn.ModuleList([self._building_block(512) for _ in range(4)])

        self.conv3 = nn.Conv2d(512, 1024, kernel_size=(1, 1), stride=(2, 2))
        self.block3 = nn.ModuleList([self._building_block(1024) for _ in range(6)])

        self.conv4 = nn.Conv2d(1024, 2048, kernel_size=(1, 1), stride=(2, 2))
        self.block4 = nn.ModuleList([self._building_block(2048) for _ in range(3)])

        self.avg_pool = GlobalAvgPool2d()
        self.fc1 = nn.Linear(2048, 1000)
        self.fc2 = nn.Linear(1000, 100)
        self.out = nn.Linear(100, 9)

    @profile
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
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

    def _building_block(self, channel_out, channel_in=None):
        if channel_in is None:
            channel_in = channel_out
        return ResBlock(channel_in, channel_out)


class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()

        self.conv1 = nn.Conv2d(channel_in, channel_out, kernel_size=(3, 3), padding=1)
        self.bn1 = nn.BatchNorm2d(channel_out)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(channel_out, channel_out, kernel_size=(3, 3), padding=1)
        self.bn2 = nn.BatchNorm2d(channel_out)
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu2 = nn.ReLU()

    @profile
    def forward(self, x):
        h = self.conv1(x)
        h = self.bn1(h)
        h = self.relu1(h)
        h = self.conv2(h)
        h = self.bn2(h)
        shortcut = self.shortcut(x)
        y = self.relu2(h + shortcut)

        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


class AttentionBlock(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        channel = channel_in // 4
        self.conv1 = nn.Conv2d(channel_in, channel, kernel_size=(3, 3), padding=1)
        self.bn = nn.BatchNorm2d(channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel, 1, kernel_size=(3, 3), padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.conv1(x)
        h = self.bn(h)
        h = self.relu(h)
        h = self.conv2(h)
        y = self.sig(h)

        return y


class SEBlock(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.fc1 = nn.Linear(channel_in, channel_in // 4)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(channel_in // 4, channel_in)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.avg_pool(x)
        h = self.fc1(h)
        h = self.relu(h)
        h = self.fc2(h)
        y = self.sig(h)

        return y


class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))
