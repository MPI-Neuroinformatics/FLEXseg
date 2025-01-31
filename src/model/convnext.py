#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 09:20:05 2024

@author: jsteiglechner
"""

from typing import List

import torch
import torch.nn as nn

from model.model_utils import (
    define_normalization,
    depthwise_conv7x7x7,
    pointwise_conv1x1x1,
)


class ConvNeXtBlock(nn.Module):
    """3D ConvNeXt Block.

    Changes:
    - use Dropout after activation instead of stochastic depth for
        regularization.
    - use composed 7x7x7 convolution.
    - use different normalization methods.
    - no permutation of axis.
    - use einsum for multiplication.

    Further reading:
    (1) A ConvNet for the 2020s (Liu et al., 2022)
            https://arxiv.org/abs/2201.03545

    Args:
        num_channels (int): Number of input channels.
    """

    expansion = 4

    def __init__(
            self,
            num_channels: int,
            dropout_rate: float = 0.,
            layer_scale_init_value: float = 1e-6,
            normalization_method="layer",
    ):
        super().__init__()
        self.depthwise_conv = depthwise_conv7x7x7(
            num_channels,
            num_channels,
            composed=True,
        )
        self.normalization = define_normalization(
            num_channels, normalization_method)
        self.pointwise_conv1 = pointwise_conv1x1x1(
            num_channels,
            self.expansion * num_channels,
        )
        self.activation = nn.GELU()
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.pointwise_conv2 = pointwise_conv1x1x1(
            self.expansion * num_channels,
            num_channels,
        )
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((num_channels)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x):
        identity_shortcut = x

        x = self.depthwise_conv(x)
        x = self.normalization(x)
        x = self.pointwise_conv1(x)

        x = self.activation(x)
        x = self.drop_out(x)
        x = self.pointwise_conv2(x)
        if self.gamma is not None:
            x = torch.einsum('c,bchwd->bchwd', self.gamma, x)

        x = identity_shortcut + x
        return x


def up_conv(
        num_channels_in: int,
        num_channels_out: int,
        scale_factor: int = 2,
        mode: str = 'transpose',
):
    """Upsampling of feature maps."""
    if mode == "trilinear":
        up_conv = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode=mode,
                align_corners=False,
            ),
            nn.Conv3d(
                in_channels=num_channels_in,
                out_channels=num_channels_out,
                kernel_size=1,
                padding=0,
                stride=1,
            ),
        )
    elif mode == 'transpose':
        up_conv = nn.ConvTranspose3d(
            in_channels=num_channels_in,
            out_channels=num_channels_out,
            kernel_size=scale_factor,
            stride=scale_factor,
        )
    elif mode == 'nearest':
        up_conv = nn.Sequential(
            nn.Upsample(
                scale_factor=scale_factor,
                mode='nearest',
            ),
            nn.Conv3d(
                in_channels=num_channels_in,
                out_channels=num_channels_out,
                kernel_size=1,
                padding=0,
                stride=1,
            ))
    else:
        raise NotImplementedError(f"Mode {mode} upsampling not implemented.")

    return up_conv


class ConvNeXtEncoder(nn.Module):
    """
    Difference to original ConvNeXt:
        - different stem like ResNet to use an additional skipping connection
        - Use of changed block
    """

    def __init__(
            self,
            input_channels: int,
            conv_block: nn.Module,
            num_blocks: List[int],
            num_channels_init: int,
            num_channels_stage: List[int],
            max_dropout_rate: float = 0.,
            layer_scale_init_value: float = 1e-6,
            normalization_method: str = "layer",
    ):
        super(ConvNeXtEncoder, self).__init__()

        # Stem motivated by resnet
        self.activation = nn.GELU()

        self.conv_stem = nn.Conv3d(
            in_channels=input_channels,
            out_channels=num_channels_init,
            kernel_size=4,
            padding=1,
            stride=2,
        )
        self.norm_stem = define_normalization(
            num_channels_init, normalization_method)

        # second downsampling. Originally in resnet max_pool
        self.conv_stem_2 = nn.Conv3d(
            in_channels=num_channels_init,
            out_channels=num_channels_stage[0],
            kernel_size=4,
            padding=1,
            stride=2,
        )
        self.norm_stem_2 = define_normalization(
            num_channels_stage[0], normalization_method)

        # Usual ConvNeXt blocks
        self.downsample_layers = nn.ModuleList()
        for i in range(3):
            downsample_layer = nn.Sequential(
                define_normalization(
                    num_channels_stage[i], normalization_method),
                nn.Conv3d(
                    num_channels_stage[i],
                    num_channels_stage[i+1],
                    kernel_size=2,
                    stride=2,
                ),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()
        # Dropout rates should increase as deeper we get
        dropout_rates = [x.item() for x in torch.linspace(
            0, max_dropout_rate, sum(num_blocks))]

        for i in range(len(num_blocks)):
            stage = nn.Sequential(
                *[
                    conv_block(
                        num_channels=num_channels_stage[i],
                        dropout_rate=dropout_rates[sum(num_blocks[:i]) + j],
                        layer_scale_init_value=layer_scale_init_value,
                        normalization_method=normalization_method,
                    ) for j in range(num_blocks[i])
                ]
            )
            self.stages.append(stage)

    def forward(self, x):
        feature_stages = []

        for i in range(4):
            # Downsampling
            if i < 1:
                # stem: scale factor 1/4
                x = self.conv_stem(x)
                feature_stages.append(x)
                x = self.norm_stem(x)
                x = self.activation(x)
                # x = self.max_pool(x)
                x = self.conv_stem_2(x)
                x = self.norm_stem_2(x)
                x = self.activation(x)
            else:
                x = self.downsample_layers[i-1](x)

            x = self.stages[i](x)
            feature_stages.append(x)

        return feature_stages


class UpConvNeXtBlock(nn.Module):
    expansion = 4

    def __init__(
            self,
            num_channels_in: int,
            num_channels_skip: int,
            num_channels_out: int,
            layer_scale_init_value: float = 1e-6,
            normalization_method="instance",
    ):
        super().__init__()
        self.upsample_layer = up_conv(num_channels_in, num_channels_in)
        self.normalization = define_normalization(
            num_channels_in + num_channels_skip, normalization_method)

        self.pointwise_conv1 = pointwise_conv1x1x1(
            num_channels_in + num_channels_skip,
            self.expansion * num_channels_out,
        )
        self.activation = nn.GELU()

        self.pointwise_conv2 = pointwise_conv1x1x1(
            self.expansion * num_channels_out,
            num_channels_out,
        )
        self.gamma = nn.Parameter(
            layer_scale_init_value * torch.ones((num_channels_out)),
            requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x, skipped_layer):
        x = self.upsample_layer(x)
        x = torch.cat((skipped_layer, x), dim=1)
        x = self.normalization(x)
        x = self.pointwise_conv1(x)

        x = self.activation(x)
        x = self.pointwise_conv2(x)
        if self.gamma is not None:
            x = torch.einsum('c,bchwd->bchwd', self.gamma, x)

        return x


class ConvNeXtDecoder(nn.Module):
    def __init__(
            self,
            up_block: nn.Module,
            conv_block: nn.Module,
            num_channels_stage: List[int],
            num_blocks: List[int] = [3, 3, 3, 3],
            max_dropout_rate: float = 0.,
            layer_scale_init_value: float = 1e-6,
            normalization_method: str = "layer",
    ):
        super(ConvNeXtDecoder, self).__init__()

        # Upsampling and skipping connections
        self.upsample_layers = nn.ModuleList()
        self.upsample_layers.append(
            up_block(
                num_channels_in=num_channels_stage[-1],
                num_channels_skip=num_channels_stage[-2],
                num_channels_out=num_channels_stage[-1],
                layer_scale_init_value=layer_scale_init_value,
                normalization_method=normalization_method,
            )
        )
        for i in range(1, len(num_channels_stage) - 1):
            self.upsample_layers.append(
                up_block(
                    num_channels_in=num_channels_stage[-i],
                    num_channels_skip=num_channels_stage[-i-2],
                    num_channels_out=num_channels_stage[-i-1],
                    layer_scale_init_value=layer_scale_init_value,
                    normalization_method=normalization_method,
                )
            )
        self.upsample_layers.append(
            up_conv(num_channels_stage[1], num_channels_stage[0])
        )

        # stages blocks
        self.stages = nn.ModuleList()
        dropout_rates = [x.item() for x in torch.linspace(
            0, max_dropout_rate, sum(num_blocks))][::-1]
        # dropout_rates = [max_dropout_rate] * sum(num_blocks)
        for i in range(len(num_blocks)):
            stage = nn.Sequential(
                *[
                    conv_block(
                        num_channels=num_channels_stage[-i-1],
                        dropout_rate=dropout_rates[sum(num_blocks[:i]) + j],
                        layer_scale_init_value=layer_scale_init_value,
                        normalization_method=normalization_method,
                    ) for j in range(num_blocks[i])
                ]
            )
            self.stages.append(stage)

    def forward(self, feature_stages):
        x = feature_stages[-1]

        for i in range(len(feature_stages) - 1):
            x = self.upsample_layers[i](
                x,
                feature_stages[-i-2]
            )

            # stages / block repetition
            x = self.stages[i](x)

        # final upsampling
        x = self.upsample_layers[-1](x)

        return x


class ConvUNeXt(nn.Module):
    def __init__(
            self,
            input_channels: int,
            output_channels: int,
            # num_planes: int = 64,
            num_blocks: List[int] = [3, 3, 9, 3],
            num_channels_stage: List[int] = [48, 96, 192, 384, 768],
            num_blocks_decoder: List[int] = [3, 9, 3, 3],
            max_dropout_rate: float = .0,
            layer_scale_init_value: float = 1e-6,
            encoder_normalization: str = "batch",
            decoder_normalitation: str = "instance",
    ):
        super(ConvUNeXt, self).__init__()

        self.encoder = ConvNeXtEncoder(
            input_channels=input_channels,
            conv_block=ConvNeXtBlock,
            num_blocks=num_blocks,
            num_channels_init=num_channels_stage[0],
            num_channels_stage=num_channels_stage[1:],
            max_dropout_rate=max_dropout_rate,
            layer_scale_init_value=layer_scale_init_value,
            normalization_method=encoder_normalization,
        )
        self.decoder = ConvNeXtDecoder(
            up_block=UpConvNeXtBlock,
            conv_block=ConvNeXtBlock,
            num_channels_stage=num_channels_stage,
            num_blocks=num_blocks_decoder,
            max_dropout_rate=max_dropout_rate,
            layer_scale_init_value=layer_scale_init_value,
            normalization_method=decoder_normalitation,
        )
        self.conv_final = nn.Conv3d(
            in_channels=num_channels_stage[0],
            out_channels=output_channels,
            kernel_size=1)

    def _init_weights(self, module):
        """Initialize model weights with He mode."""
        if isinstance(module, (nn.ConvTranspose3d, nn.Conv3d, nn.Linear)):
            nn.init.trunc_normal_(module.weight.data, std=.2)
            if module.bias is not None:
                nn.init.constant_(module.bias.data, 0)

    def forward(self, x):
        x = self.encoder(x)  # feature stages
        x = self.decoder(x)

        output_img = self.conv_final(x)

        return output_img
