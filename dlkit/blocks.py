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

import torch.nn as nn
import torch.nn.functional as F


# ## Residual Block

class ResBlock(nn.Module):
    def __init__(self, channel_in, channel_out):
        super().__init__()
        channel = channel_out // 4
        self.cbr1 = CBR(channel_in, channel, kernel_size=1)
        self.cbr2 = CBR(channel, channel, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(channel, channel_out, kernel_size=1, padding=0)
        self.bn = nn.BatchNorm2d(channel_out)
        self.shortcut = self._shortcut(channel_in, channel_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        h = self.cbr1(x)
        h = self.cbr2(h)
        h = self.conv(h)
        h = self.bn(h)
        shortcut = self.shortcut(x)
        y = self.relu(h + shortcut)

        return y

    def _shortcut(self, channel_in, channel_out):
        if channel_in != channel_out:
            return self._projection(channel_in, channel_out)
        else:
            return lambda x: x

    def _projection(self, channel_in, channel_out):
        return nn.Conv2d(channel_in, channel_out, kernel_size=(1, 1), padding=0)


# ## Attention Block

class AttentionBlock(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        channel = channel_in // 4
        self.cbr = CBR(channel_in, channel, kernel_size=3, padding=1)
        self.conv = nn.Conv2d(channel, 1, kernel_size=3, padding=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.cbr(x)
        h = self.conv(h)
        y = self.sig(h * x)

        return y


# ## Squeeze & Excitation Block

class SEBlock(nn.Module):
    def __init__(self, channel_in):
        super().__init__()
        self.avg_pool = GlobalAvgPool2d()
        self.conv1 = nn.Conv2d(channel_in, channel_in // 4, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(channel_in // 4, channel_in, kernel_size=1)
        self.sig = nn.Sigmoid()

    def forward(self, x):
        h = self.avg_pool(x)
        h = self.conv1(h)
        h = self.relu(h)
        h = self.conv2(h)
        y = self.sig(h * x)

        return y


# ## Global Average Pooling

class GlobalAvgPool2d(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return F.avg_pool2d(x, kernel_size=x.size()[2:]).view(-1, x.size(1))


# ## Convolution + Batch Normalization + ReLU

class CBR(nn.Module):
    def __init__(self, channel_in, channel_out, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            channel_in,
            channel_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
        )
        self.bn = nn.BatchNorm2d(channel_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x
