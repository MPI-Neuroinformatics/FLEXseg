"""The module provides building blocks for UNet.

Further reading: https://arxiv.org/abs/1505.04597
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import activations


def conv_block_3d(num_channels_in,
                  num_channels_out,
                  kernel_size=3,
                  activation='relu',
                  dropout=0.2,
                  batchnorm=True):
    """Convolve the input twice."""
    # first layer
    layers = [
        nn.Conv3d(in_channels=num_channels_in,
                  out_channels=num_channels_out,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=(not batchnorm))
    ]
    if batchnorm:
        layers.append(nn.BatchNorm3d(num_features=num_channels_out))
    layers.append(activations[activation])
    # Dropout for regularization
    if dropout != None:
        layers.append(nn.Dropout(p=dropout))

    # second layer
    layers.append(
        nn.Conv3d(in_channels=num_channels_out,
                  out_channels=num_channels_out,
                  kernel_size=kernel_size,
                  stride=1,
                  padding=1,
                  dilation=1,
                  groups=1,
                  bias=(not batchnorm)))
    if batchnorm:
        layers.append(nn.BatchNorm3d(num_features=num_channels_out))
    layers.append(activations[activation])

    return nn.Sequential(*layers)  # asterix decompose list of variables


class Interpolate(nn.Module):
    """Interpolation for upconv path."""

    def __init__(self, scale_factor, mode, size=None):
        super(Interpolate, self).__init__()
        self.interp = F.interpolate
        self.scale_factor = scale_factor
        self.mode = mode
        self.size = size

    def forward(self, x):
        x = self.interp(x,
                        size=self.size,
                        scale_factor=self.scale_factor,
                        mode=self.mode)
        return x


def upconv_block_3d(num_channels_in, num_channels_out, mode='transpose'):
    """Deconvolve the input once."""
    if mode == 'transpose':
        layers = [
            nn.ConvTranspose3d(in_channels=num_channels_in,
                               out_channels=num_channels_out,
                               kernel_size=2,
                               stride=2)
        ]
    elif mode == 'trilinear':
        layers = [
            Interpolate(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels=num_channels_in,
                      out_channels=num_channels_out,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1)
        ]
    else:
        layers = [
            Interpolate(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels=num_channels_in,
                      out_channels=num_channels_out,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1)
        ]

    return nn.Sequential(*layers)


class UNetEncoder(nn.Module):
    """Encoder for UNet."""

    def __init__(self,
                 input_channels,
                 num_init_channels,
                 encoder_depth=5,
                 batchnorm=True):
        super(UNetEncoder, self).__init__()
        self.layers = nn.ModuleList([
            conv_block_3d(num_channels_in=input_channels,
                          num_channels_out=num_init_channels,
                          batchnorm=batchnorm)
        ])
        for i in range(1, encoder_depth):
            self.layers.append(nn.MaxPool3d(kernel_size=2, stride=2))
            self.layers.append(
                conv_block_3d(num_channels_in=num_init_channels * 2**(i - 1),
                              num_channels_out=num_init_channels * 2**i,
                              batchnorm=batchnorm))

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            if torch.typename(
                    layer) == 'torch.nn.modules.container.Sequential':
                skip_connections.append(x)

        return x, skip_connections


class UNetDecoder(nn.Module):
    """Decoder for UNet."""

    def __init__(self, num_init_channels, decoder_depth=4, batchnorm=True):
        super(UNetDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleList([
                upconv_block_3d(num_channels_in=num_init_channels * 2**i,
                                num_channels_out=(num_init_channels *
                                                  2**(i - 1))),
                conv_block_3d(num_channels_in=num_init_channels * 2**i,
                              num_channels_out=num_init_channels * 2**(i - 1),
                              batchnorm=batchnorm)
            ]) for i in range(decoder_depth, 0, -1)
        ])

    def forward(self, x, skipped_layers):
        for i in range(len(self.layers)):
            up = self.layers[i][0](x)
            merge = torch.cat((skipped_layers[-(i + 1)], up), dim=1)
            x = self.layers[i][1](merge)

        return x


class UNet(nn.Module):
    """UNet."""

    def __init__(self,
                 input_channels,
                 output_channels,
                 num_planes=64,
                 depth=5,
                 batchnorm=True):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597

        Modifications to the original paper:
        (1) padding is used in 3x3 convolutions to prevent loss
            of border pixels
        (2) batchnorm is usually False.
        """
        super(UNet, self).__init__()
        self.encoder = UNetEncoder(input_channels=input_channels,
                                   num_init_channels=num_planes,
                                   encoder_depth=depth,
                                   batchnorm=batchnorm)
        self.decoder = UNetDecoder(num_init_channels=num_planes,
                                   decoder_depth=depth - 1,
                                   batchnorm=batchnorm)
        self.conv_final = nn.Conv3d(in_channels=num_planes,
                                    out_channels=output_channels,
                                    kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections[:-1])

        output_img = self.conv_final(x)

        return output_img
