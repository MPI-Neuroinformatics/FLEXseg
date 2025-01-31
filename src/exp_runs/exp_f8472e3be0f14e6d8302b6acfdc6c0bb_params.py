#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:22:35 2024

@author: jsteiglechner

As presented in SNS Tübingen 2024.
"""

import numpy as np

from model.resnet import ResMirror


EXPERIMENT_ID = "f8472e3be0f14e6d8302b6acfdc6c0bb"


MODEL_WEIGHTS_RFP = f"model_weights/{EXPERIMENT_ID}_segmentation.pt"
NUM_TYPES = 5
# Types = {
#   "0": "background",
#   "1": "cortical_gm_left",
#   "2": "cortical_gm_right",
#   "3": "cerebrum_left",
#   "4": "cerebrum_right",
# }
MODEL_INPUT_SHAPE = [128, 128, 128]
STANDARDIZATION_SHAPE = [384, 384, 384]

generator = ResMirror(
    input_channels=1,
    output_channels=NUM_TYPES,
    num_planes=48,
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

    # standardize
    img_numel = np.prod(image.shape)
    norm_numel = np.prod(STANDARDIZATION_SHAPE)

    mean = np.mean(np.append(
        image.flatten(),
        np.zeros(norm_numel - img_numel),
    ))
    std = np.std(np.append(
        image.flatten(),
        np.zeros(norm_numel - img_numel),
    ))
    image = np.divide(
        np.subtract(image, mean, out=image),
        std,
        out=image
    ).astype(np.float32)

    return image, image_affine
