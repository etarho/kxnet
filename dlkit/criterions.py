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


class Criterion:

    def __init__(self, mode):

        self.mode = mode

        if mode == 'mae':
            self.criterion = nn.L1Loss()

        if mode == 'mse':
            self.criterion = nn.MSELoss()

        if mode == 'huber':
            self.criterion = nn.SmoothL1Loss()

        if mode == 'cos':
            self.criterion = nn.CosineSimilarity()

    def __call__(self, pred, target):
        if self.mode == 'cos':
            return 1 - self.criterion(pred, target).mean()

        else:
            return self.criterion(pred, target)
