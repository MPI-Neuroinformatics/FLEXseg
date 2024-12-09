#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:22:35 2024

@author: jsteiglechner
"""

import numpy as np

from image_operations.feature_scaling import quantile_clipping
from model.resnet import ResMirror


# experiment with higher adversarial lambda
EXPERIMENT_ID = "9d4f375c2bdc4b2fbc02bdd399fcd025"

MODEL_WEIGHTS_RFP = f"model_weights/{EXPERIMENT_ID}_segmentation_state_dict.pth"
NUM_TYPES = 6
# Types = {
#   "0": "background",
#   "1": "cortical_gm_left",
#   "2": "cortical_gm_right",
#   "3": "cerebrum_left",
#   "4": "cerebrum_right",
#   "5": "intracranial_volume"
# }
MODEL_INPUT_SHAPE = [128, 128, 128]
MIN_SHAPE = [256, 256, 256]
MAX_SHAPE = [512, 512, 512]

generator = ResMirror(
    input_channels=1,
    output_channels=NUM_TYPES,
    num_planes=64,
    name_block="bottleneck",
    num_blocks=[3, 4, 23, 3],
    activation="relu",
    variant="original",
)


def preprocessing(image: np.ndarray, image_affine: np.ndarray):
    """
    Preprocessing that was done in the experiment.

    Parameters
    ----------
    image : np.ndarray
        DESCRIPTION.
    image_affine : np.ndarray
        DESCRIPTION.

    Raises
    ------
    ValueError
        DESCRIPTION.

    Returns
    -------
    image_clipped : TYPE
        DESCRIPTION.
    image_affine : TYPE
        DESCRIPTION.

    """

    # clipping
    image_clipped = quantile_clipping(
        image, lower_quantile=0.01, upper_quantile=0.99)

    # padding to minimal shape
    if np.any(np.array(image_clipped.shape) < np.array(MIN_SHAPE)):
        size_diff = np.subtract(MIN_SHAPE, image_clipped.shape)
        size_diff = np.where(size_diff < 0, 0, size_diff)
        pad_widths_before = ((size_diff - np.mod(size_diff, 2)) /
                             2).astype(int)
        pad_widths_after = size_diff - pad_widths_before
        pad_width = tuple(zip(pad_widths_before, pad_widths_after))

        image_clipped = np.pad(
            image_clipped,
            pad_width,
            constant_values=np.min(image_clipped),
        )
        image_affine[:3, 3] -= image_affine[:3, :3] @ pad_widths_before

    # cropping to maximal shape
    if np.any(np.array(image_clipped.shape) > np.array(MAX_SHAPE)):
        size_diff = np.subtract(image_clipped.shape, MAX_SHAPE)
        crop_start = ((size_diff - np.mod(size_diff, 2)) / 2).astype(int)
        crop_start = np.max([crop_start, [0, 0, 0]], axis=0)
        size = tuple(np.min([MAX_SHAPE, image_clipped.shape], axis=0))

        if size > image.shape:
            raise ValueError(
                f'Target size {size} is not allowed to be larger than image {image.shape}.'
            )
        crop_end = np.add(crop_start, size)

        image_clipped = image_clipped[
            crop_start[0]:crop_end[0],
            crop_start[1]:crop_end[1],
            crop_start[2]:crop_end[2],
        ]
        image_affine[:3, 3] += image_affine[:3, :3] @ crop_start

    # standardization
    mean = np.mean(image_clipped)
    std = np.std(image_clipped)
    image_clipped = np.divide(np.subtract(
        image_clipped, mean, out=image_clipped), std, out=image_clipped)

    return image_clipped, image_affine
