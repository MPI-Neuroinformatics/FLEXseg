"""This module provides activation functions for neurons."""

import torch
import torch.nn as nn


class Swish(nn.Module):
    """Swish Activation Function: https://arxiv.org/abs/1710.05941."""

    def __init__(self, beta):
        super(Swish, self).__init__()
        self.beta = beta

    def forward(self, x):
        return torch.mul(torch.sigmoid(torch.mul(self.beta, x)), x)


activations = nn.ModuleDict([['lrelu', nn.LeakyReLU(inplace=True)],
                             ['relu', nn.ReLU(inplace=True)],
                             ['swish', Swish(beta=1)]])
