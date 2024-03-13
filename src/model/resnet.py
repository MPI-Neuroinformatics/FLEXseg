"""This module provides building blocks of ResNet for machine learning."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.activation_functions import activations


def conv3x3x3(num_planes_in, num_planes_out, stride=1):
    """3x3x3 convolution with padding."""
    return nn.Conv3d(in_channels=num_planes_in,
                     out_channels=num_planes_out,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1x1x1(num_planes_in, num_planes_out, stride=1):
    """1x1x1 convolution."""
    return nn.Conv3d(in_channels=num_planes_in,
                     out_channels=num_planes_out,
                     kernel_size=1,
                     stride=stride,
                     padding=0,
                     bias=False)


class Interpolate(nn.Module):
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


def upconv(num_planes_in, num_planes_out, mode='transpose'):
    """Upconvolce."""
    if mode == 'transpose':
        return nn.ConvTranspose3d(in_channels=num_planes_in,
                                  out_channels=num_planes_out,
                                  kernel_size=2,
                                  stride=2)
    elif mode == 'trilinear':
        return nn.Sequential(
            Interpolate(scale_factor=2, mode='trilinear'),
            nn.Conv3d(in_channels=num_planes_in,
                      out_channels=num_planes_out,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1))
    else:
        return nn.Sequential(
            Interpolate(scale_factor=2, mode='nearest'),
            nn.Conv3d(in_channels=num_planes_in,
                      out_channels=num_planes_out,
                      kernel_size=1,
                      padding=0,
                      stride=1,
                      groups=1))


class ResidualBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 num_planes_in,
                 num_planes_out,
                 stride=1,
                 activation='relu',
                 downsample=None,
                 batchnorm=True,
                 variant='original',
                 dropout_rate=0.2):
        super(ResidualBlock, self).__init__()
        # Both self.conv1 and self.downsample layers downsample the input when
        # stride != 1
        assert variant in ['original', 'pre_activation']
        self.batchnorm = batchnorm
        self.variant = variant

        if self.batchnorm:
            if self.variant == 'pre_activation':
                # BatchNormalization and Activation before Convolution
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_in)
            elif self.variant == 'original':
                # Convolution before BatchNormalization and Activation
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_out)

        self.conv1 = conv3x3x3(num_planes_in, num_planes_out, stride)
        self.drop_out = nn.Dropout(p=dropout_rate)
        self.conv2 = conv3x3x3(num_planes_out, num_planes_out * self.expansion)
        if self.batchnorm:
            self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
        self.activation = activations[activation]
        self.downsample = downsample

    def forward(self, x):
        identity_shortcut = x

        if self.variant == 'original':
            out = self.conv1(x)
            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            if self.batchnorm:
                out = self.bn2(out)

            if self.downsample is not None:
                identity_shortcut = self.downsample(x)

            out = torch.add(identity_shortcut, out)
            out = self.activation(out)

        elif self.variant == 'pre_activation':
            if self.batchnorm:
                out = self.bn1(x)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.drop_out(out)

            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            if self.downsample is not None:
                identity_shortcut = self.downsample(x)

            out = torch.add(identity_shortcut, out)

        return out


class BottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 num_planes_in,
                 num_planes_out,
                 stride=1,
                 activation='relu',
                 downsample=None,
                 batchnorm=True,
                 variant='original',
                 dropout_rate=0.2):
        super(BottleneckBlock, self).__init__()
        assert variant in ['original', 'pre_activation']
        # The 1x1x1 layers are responsible for reducing and then restoring
        # (expanding) dimensions, leaving the 3x3x3 layer a bottleneck
        self.variant = variant
        self.batchnorm = batchnorm

        if self.variant == 'original':
            self.conv1 = conv1x1x1(num_planes_in, num_planes_out)
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv2 = conv3x3x3(num_planes_out, num_planes_out, stride)
            if self.batchnorm:
                self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv3 = conv1x1x1(num_planes_out,
                                   num_planes_out * self.expansion)
            if self.batchnorm:
                self.bn3 = nn.BatchNorm3d(num_features=(num_planes_out *
                                                        self.expansion))
        elif self.variant == 'pre_activation':
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_in)
            self.conv1 = conv1x1x1(num_planes_in, num_planes_out)

            self.drop_out = nn.Dropout(p=dropout_rate)

            if self.batchnorm:
                self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv2 = conv3x3x3(num_planes_out, num_planes_out, stride)
            if self.batchnorm:
                self.bn3 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv3 = conv1x1x1(num_planes_out,
                                   num_planes_out * self.expansion)
        self.activation = activations[activation]
        self.downsample = downsample

    def forward(self, x):
        identity_shortcut = x

        if self.variant == 'original':
            out = self.conv1(x)
            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            if self.batchnorm:
                out = self.bn3(out)

            if self.downsample is not None:
                identity_shortcut = self.downsample(x)

            out = torch.add(identity_shortcut, out)
            out = self.activation(out)

        elif self.variant == 'pre_activation':
            if self.batchnorm:
                out = self.bn1(x)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.drop_out(out)

            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            if self.batchnorm:
                out = self.bn3(out)
            out = self.activation(out)
            out = self.conv3(out)

            if self.downsample is not None:
                identity_shortcut = self.downsample(x)

            out = torch.add(identity_shortcut, out)

        return out


class UpResidualBlock(nn.Module):
    expansion = 1

    def __init__(self,
                 num_planes_in,
                 num_planes_out,
                 activation='relu',
                 upsample=None,
                 batchnorm=True,
                 variant='original',
                 dropout_rate=0.2):
        super(UpResidualBlock, self).__init__()
        assert variant in ['original', 'pre_activation']
        self.variant = variant
        self.batchnorm = batchnorm

        if self.variant == 'pre_activation':
            # BatchNormalization and Activation before Convolution
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_in)
            self.drop_out = nn.Dropout(p=dropout_rate)
        elif self.variant == 'original':
            # Convolution before BatchNormalization and Activation
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_out)

        self.up = upconv(num_planes_in, num_planes_in)
        self.conv1 = conv3x3x3(num_planes_in, num_planes_out)
        self.conv2 = conv3x3x3(num_planes_out, num_planes_out)
        if self.batchnorm:
            self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
        self.activation = activations[activation]
        self.upsample = upsample

    def forward(self, x):
        identity_shortcut = x

        if self.variant == 'original':

            out = self.up(x)
            out = self.conv1(out)
            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            if self.batchnorm:
                out = self.bn2(out)

            if self.upsample is not None:
                identity_shortcut = self.upsample(x)

            out = torch.add(identity_shortcut, out)
            out = self.activation(out)

        elif self.variant == 'pre_activation':
            out = self.up(x)

            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.drop_out(out)

            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            if self.upsample is not None:
                identity_shortcut = self.upsample(x)

            out = torch.add(identity_shortcut, out)

        return out


class UpBottleneckBlock(nn.Module):
    expansion = 4

    def __init__(self,
                 num_planes_in,
                 num_planes_out,
                 stride=1,
                 activation='relu',
                 upsample=None,
                 batchnorm=True,
                 variant='original',
                 dropout_rate=0.2):
        super(UpBottleneckBlock, self).__init__()
        assert variant in ['original', 'pre_activation']
        # The 1x1x1 layers are responsible for reducing and then restoring
        # (expanding) dimensions, leaving the 3x3x3 layer a bottleneck
        self.variant = variant
        self.batchnorm = batchnorm

        self.up = upconv(num_planes_in, num_planes_in)
        if self.variant == 'original':
            self.conv1 = conv1x1x1(num_planes_in, num_planes_out)
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv2 = conv3x3x3(num_planes_out, num_planes_out, stride)
            if self.batchnorm:
                self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv3 = conv1x1x1(num_planes_out,
                                   num_planes_out * self.expansion)
            if self.batchnorm:
                self.bn3 = nn.BatchNorm3d(num_features=(num_planes_out *
                                                        self.expansion))
        elif self.variant == 'pre_activation':
            if self.batchnorm:
                self.bn1 = nn.BatchNorm3d(num_features=num_planes_in)
            self.conv1 = conv1x1x1(num_planes_in, num_planes_out)
            if self.batchnorm:
                self.bn2 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv2 = conv3x3x3(num_planes_out, num_planes_out, stride)
            if self.batchnorm:
                self.bn3 = nn.BatchNorm3d(num_features=num_planes_out)
            self.conv3 = conv1x1x1(num_planes_out,
                                   num_planes_out * self.expansion)
            self.drop_out = nn.Dropout(p=dropout_rate)
        self.activation = activations[activation]
        self.upsample = upsample

    def forward(self, x):
        identity_shortcut = x

        out = self.up(x)

        if self.variant == 'original':
            out = self.conv1(out)
            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)

            out = self.conv2(out)
            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)

            out = self.conv3(out)
            if self.batchnorm:
                out = self.bn3(out)

            if self.upsample is not None:
                identity_shortcut = self.upsample(x)

            out = torch.add(identity_shortcut, out)
            out = self.activation(out)

        elif self.variant == 'pre_activation':
            if self.batchnorm:
                out = self.bn1(out)
            out = self.activation(out)
            out = self.conv1(out)

            out = self.drop_out(out)

            if self.batchnorm:
                out = self.bn2(out)
            out = self.activation(out)
            out = self.conv2(out)

            if self.batchnorm:
                out = self.bn3(out)
            out = self.activation(out)
            out = self.conv3(out)

            if self.upsample is not None:
                identity_shortcut = self.upsample(x)

            out = torch.add(identity_shortcut, out)

        return out


class ResNet(nn.Module):
    def __init__(self,
                 block,
                 num_blocks,
                 input_channels,
                 output_channels,
                 num_planes=64,
                 batchnorm=True,
                 variant='original',
                 activation='relu'):
        super(ResNet, self).__init__()
        self.num_planes_in = num_planes
        self.variant = variant
        self.batchnorm = batchnorm
        self.conv1 = nn.Conv3d(in_channels=input_channels,
                               out_channels=self.num_planes_in,
                               kernel_size=7,
                               stride=2,
                               padding=3,
                               bias=False)
        if self.batchnorm:
            self.bn1 = nn.BatchNorm3d(self.num_planes_in)
        self.activation = activations[activation]
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_down_layer(block,
                                            num_planes,
                                            num_blocks[0],
                                            activation=activation,
                                            batchnorm=self.batchnorm,
                                            variant=self.variant)
        self.layer2 = self._make_down_layer(block,
                                            num_planes * 2,
                                            num_blocks[1],
                                            stride=2,
                                            activation=activation,
                                            batchnorm=self.batchnorm,
                                            variant=self.variant)
        self.layer3 = self._make_down_layer(block,
                                            num_planes * 4,
                                            num_blocks[2],
                                            stride=2,
                                            activation=activation,
                                            batchnorm=self.batchnorm,
                                            variant=self.variant)
        self.layer4 = self._make_down_layer(block,
                                            num_planes * 8,
                                            num_blocks[3],
                                            stride=2,
                                            activation=activation,
                                            batchnorm=self.batchnorm,
                                            variant=self.variant)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(num_planes * 2**3 * block.expansion,
                            output_channels)

        # Initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight,
                                        mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_down_layer(self,
                         block,
                         num_planes_out,
                         num_blocks,
                         stride=1,
                         activation='relu',
                         batchnorm=True,
                         variant='original'):
        downsample = None
        if stride != 1 or self.num_planes_in != (num_planes_out *
                                                 block.expansion):
            downsample = [
                conv1x1x1(self.num_planes_in,
                          num_planes_out * block.expansion,
                          stride=stride), ]
            if batchnorm:
                downsample.append(
                    nn.BatchNorm3d(num_planes_out * block.expansion))
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(num_planes_in=self.num_planes_in,
                  num_planes_out=num_planes_out,
                  stride=stride,
                  downsample=downsample,
                  activation=activation,
                  batchnorm=batchnorm,
                  variant=variant))
        self.num_planes_in = num_planes_out * block.expansion
        for _ in range(1, num_blocks):
            layers.append(
                block(num_planes_in=self.num_planes_in,
                      num_planes_out=num_planes_out,
                      activation=activation,
                      batchnorm=batchnorm,
                      variant=variant))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        if self.batchnorm:
            x = self.bn1(x)
        x = self.activation(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


class ResNetEncoder(nn.Module):
    def __init__(self,
                 input_channels,
                 num_planes_in,
                 block,
                 num_blocks,
                 activation='relu',
                 batchnorm=True,
                 variant='original'):
        super(ResNetEncoder, self).__init__()
        self.num_planes_in = num_planes_in
        self.layers = nn.ModuleList([
            nn.Conv3d(in_channels=input_channels,
                      out_channels=self.num_planes_in,
                      kernel_size=7,
                      stride=2,
                      padding=3,
                      bias=False),
        ])
        if batchnorm:
            self.layers.append(nn.BatchNorm3d(num_features=num_planes_in))
        self.layers.extend([
            activations[activation],
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        ])
        for i in range(len(num_blocks)):
            if i == 0:
                _stride = 1
            else:
                _stride = 2

            self.layers.append(
                self._make_down_layer(block=block,
                                      num_planes_out=num_planes_in * 2**i,
                                      num_blocks_stacked=num_blocks[i],
                                      stride=_stride,
                                      activation=activation,
                                      batchnorm=batchnorm,
                                      variant=variant))

    def _make_down_layer(self,
                         block,
                         num_planes_out,
                         num_blocks_stacked,
                         stride=1,
                         activation='relu',
                         batchnorm=True,
                         variant='original'):
        downsample = None
        if stride != 1 or self.num_planes_in != (num_planes_out *
                                                 block.expansion):
            downsample = [
                conv1x1x1(self.num_planes_in,
                          num_planes_out * block.expansion,
                          stride=stride), ]
            if batchnorm:
                downsample.append(
                    nn.BatchNorm3d(num_planes_out * block.expansion))
            downsample = nn.Sequential(*downsample)

        layers = []
        layers.append(
            block(num_planes_in=self.num_planes_in,
                  num_planes_out=num_planes_out,
                  stride=stride,
                  downsample=downsample,
                  activation=activation,
                  batchnorm=batchnorm,
                  variant=variant))
        self.num_planes_in = num_planes_out * block.expansion
        for _ in range(1, num_blocks_stacked):
            layers.append(
                block(num_planes_in=self.num_planes_in,
                      num_planes_out=num_planes_out,
                      activation=activation,
                      batchnorm=batchnorm,
                      variant=variant))

        return nn.Sequential(*layers)

    def forward(self, x):
        skip_connections = []
        for layer in self.layers:
            x = layer(x)
            if (torch.typename(layer)
                    == 'torch.nn.modules.container.Sequential' or
                    torch.typename(layer) == 'torch.nn.modules.conv.Conv3d'):
                skip_connections.append(x)

        return x, skip_connections


class ResNetDecoder(nn.Module):
    def __init__(self,
                 num_planes_in,
                 down_block,
                 up_block,
                 num_blocks,
                 activation='relu',
                 batchnorm=True,
                 variant='original'):
        super(ResNetDecoder, self).__init__()
        self.num_planes_in = num_planes_in * up_block.expansion
        self.layers = nn.ModuleList([
            self._make_up_layer(down_block=down_block,
                                up_block=up_block,
                                num_planes_out=(num_planes_in
                                                // 2**(len(num_blocks) - i)),
                                num_blocks_stacked=num_blocks[i],
                                activation=activation,
                                batchnorm=batchnorm,
                                variant=variant)
            for i in range(len(num_blocks) - 1, 0, -1)
        ])
        self.layers.append(
            self._make_up_layer(down_block=down_block,
                                up_block=up_block,
                                num_planes_out=(num_planes_in //
                                                2**(len(num_blocks) - 1)),
                                num_blocks_stacked=num_blocks[0],
                                activation=activation,
                                batchnorm=batchnorm,
                                variant=variant))
        conv_adjust = [
            conv1x1x1(
                num_planes_in // 2**(len(num_blocks) - 1) *
                (1 + up_block.expansion), num_planes_in //
                2**(len(num_blocks) - 1) * up_block.expansion),
        ]
        if batchnorm:
            conv_adjust.append(
                nn.BatchNorm3d(num_planes_in // 2**(len(num_blocks) - 1) *
                               up_block.expansion))
        self.conv_adjust = nn.Sequential(*conv_adjust)
        self.up_final = upconv(
            num_planes_in=(num_planes_in // 2**(len(num_blocks) - 1) *
                           up_block.expansion),
            num_planes_out=num_planes_in // 2**(len(num_blocks) - 1))

    def _make_up_layer(self,
                       down_block,
                       up_block,
                       num_planes_out,
                       num_blocks_stacked,
                       activation='relu',
                       batchnorm=True,
                       variant='original'):
        upsample = [
            upconv(num_planes_in=self.num_planes_in,
                   num_planes_out=self.num_planes_in),
            conv1x1x1(self.num_planes_in, num_planes_out * up_block.expansion),
        ]
        if batchnorm:
            upsample.append(nn.BatchNorm3d(num_planes_out *
                                           up_block.expansion))
        upsample = nn.Sequential(*upsample)

        layers = []
        layers.append(
            up_block(
                num_planes_in=self.num_planes_in,
                num_planes_out=num_planes_out,  # expansion in up_block
                upsample=upsample,
                activation=activation,
                batchnorm=batchnorm,
                variant=variant))
        # Case after merge with skip connection
        downsample = None
        if self.num_planes_in != (num_planes_out * up_block.expansion):
            downsample = [
                conv1x1x1(self.num_planes_in,
                          num_planes_out * down_block.expansion,
                          stride=1), ]
            if batchnorm:
                downsample.append(
                    nn.BatchNorm3d(num_planes_out * down_block.expansion))
            downsample = nn.Sequential(*downsample)
        layers.append(
            down_block(num_planes_in=self.num_planes_in,
                       num_planes_out=num_planes_out,
                       downsample=downsample,
                       activation=activation,
                       batchnorm=batchnorm,
                       variant=variant))
        self.num_planes_in = num_planes_out * up_block.expansion

        for _ in range(2, num_blocks_stacked):
            downsample = None
            if self.num_planes_in != (num_planes_out * down_block.expansion):
                downsample = [
                    conv1x1x1(self.num_planes_in,
                              num_planes_out * down_block.expansion,
                              stride=1), ]
                if batchnorm:
                    downsample.append(
                        nn.BatchNorm3d(num_planes_out * down_block.expansion))
                downsample = nn.Sequential(*downsample)
            layers.append(
                down_block(num_planes_in=self.num_planes_in,
                           num_planes_out=num_planes_out,
                           downsample=downsample,
                           activation=activation,
                           batchnorm=batchnorm,
                           variant=variant))

        return nn.ModuleList(layers)

    def forward(self, x, skipped_layers):
        for i in range(len(self.layers)):
            up = self.layers[i][0](x)
            x = torch.cat((skipped_layers[-(i + 1)], up), dim=1)

            # After last merge adjust planes because of only skipped_layer with
            # plane=1 regardless of expansion
            if i == len(self.layers) - 1:
                x = self.conv_adjust(x)

            for j in range(1, len(self.layers[i])):
                x = self.layers[i][j](x)

        x = self.up_final(x)

        return x


class ResMirror(nn.Module):
    def __init__(self,
                 input_channels,
                 output_channels,
                 num_planes=64,
                 num_blocks=[2, 2, 2, 2],
                 name_block='residual',
                 activation='relu',
                 batchnorm=True,
                 variant='original'):
        super(ResMirror, self).__init__()
        if name_block == 'residual':
            down_block = ResidualBlock
            up_block = UpResidualBlock
        elif name_block == 'bottleneck':
            down_block = BottleneckBlock
            up_block = UpBottleneckBlock

        self.encoder = ResNetEncoder(input_channels=input_channels,
                                     num_planes_in=num_planes,
                                     block=down_block,
                                     num_blocks=num_blocks,
                                     activation=activation,
                                     batchnorm=batchnorm,
                                     variant=variant)
        self.decoder = ResNetDecoder(num_planes_in=num_planes *
                                     2**(len(num_blocks) - 1),
                                     down_block=down_block,
                                     up_block=up_block,
                                     num_blocks=num_blocks,
                                     activation=activation,
                                     batchnorm=batchnorm,
                                     variant=variant)
        self.conv_final = nn.Conv3d(in_channels=num_planes,
                                    out_channels=output_channels,
                                    kernel_size=1)

    def forward(self, x):
        x, skip_connections = self.encoder(x)
        x = self.decoder(x, skip_connections[:-1])

        output_img = self.conv_final(x)

        return output_img
