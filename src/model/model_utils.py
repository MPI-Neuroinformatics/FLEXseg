"""This module provides activation functions for neurons."""

import torch
import torch.nn as nn
import torch.nn.functional as F


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


class LayerNorm(nn.Module):
    """Layer Normalization fixing the mean and the variance of the summed
    inputs within each layer.

    Notes:
        [2] Layer Normalization (Ba et al. 2016)
            https://arxiv.org/abs/1607.06450

    Args:
        normalized_shape (Tuple):
            Dimension of hidden states.

    """

    def __init__(
            self,
            normalized_shape,
            epsilon=1e-6,
            data_format="channels_last",
    ):
        super().__init__()
        # Initialize gain and bias parameters
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))

        self.epsilon = epsilon
        self.data_format = data_format
        if self.data_format not in ["channels_first", "channels_last"]:
            raise NotImplementedError(f"Data Format {data_format} missing.")
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.epsilon,
            )
        elif self.data_format == "channels_first":  # (N, C, H, W, D)
            x = x.permute(0, 2, 3, 4, 1)
            x = F.layer_norm(
                x,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.epsilon,
            )
            x = x.permute(0, 4, 1, 2, 3)
            # The papers implementation:
            # u = x.mean(1, keepdim=True)
            # s = (x - u).pow(2).mean(1, keepdim=True)
            # x = (x - u) / torch.sqrt(s + self.epsilon)
            # x = self.weight[:, None, None, None] * \
            #     x + self.bias[:, None, None, None]
            return x


def define_normalization(
        num_channels: int,
        normalization_method: str = "layer",
) -> nn.Module:
    """
    Define normalization layer.

    Parameters
    ----------
    num_channels : int
        Number of channels to normalize.
    normalization_method : str, optional
        Mathod to use for normalization. The default is "layer".

    Raises
    ------
    KeyError
        If normalization_method not available.

    Returns
    -------
    normalization_layer : nn.Module
        Normalization layer.

    """
    if normalization_method not in ["layer", "batch", "instance"]:
        raise KeyError(f"Normalization {normalization_method} not available.")

    if normalization_method == "layer":
        normalization_layer = LayerNorm(
            num_channels, epsilon=1e-6, data_format="channels_first",)
    elif normalization_method == "batch":
        normalization_layer = nn.BatchNorm3d(num_channels)
    elif normalization_method == "instance":
        normalization_layer = nn.InstanceNorm3d(num_channels)

    return normalization_layer


# -----------------------------------------------------------------------------
# Convolutional layers


def pointwise_conv1x1x1(num_channels_in, num_channels_out):
    """1x1x1 conv."""
    return nn.Conv3d(
        in_channels=num_channels_in,
        out_channels=num_channels_out,
        kernel_size=1,
        bias=True,
    )


def depthwise_conv7x7x7(
        num_channels_in, num_channels_out, stride=1, composed=False,):
    """7x7x7 convolution with padding."""

    if composed:
        return nn.Sequential(
            nn.Conv3d(
                in_channels=num_channels_in,
                out_channels=num_channels_out,
                kernel_size=3,
                stride=stride,
                padding=1,
                groups=num_channels_in,
                bias=True,
            ),
            nn.Conv3d(
                in_channels=num_channels_out,
                out_channels=num_channels_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=num_channels_in,
                bias=True,
            ),
            nn.Conv3d(
                in_channels=num_channels_out,
                out_channels=num_channels_out,
                kernel_size=3,
                stride=1,
                padding=1,
                groups=num_channels_in,
                bias=True,
            ),
        )
    else:
        return nn.Conv3d(
            in_channels=num_channels_in,
            out_channels=num_channels_out,
            kernel_size=7,
            stride=stride,
            padding=3,
            groups=num_channels_in,
            bias=True,
        )
