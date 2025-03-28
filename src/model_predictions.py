#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module provide functionality to make model predictions."
"""


import itertools
from typing import List, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from mrimage_processing.spatial_transformation.crop_image import crop_image


def distribute_regions_uniformely(
        region: List[int],
        subregion_size: List[int],
        num_subregions: Union[int, List[int]],
) -> np.ndarray:
    """Distribute regions uniformely."""
    dim_region = len(region)

    if dim_region != len(subregion_size):
        raise ValueError(
            'Dimensionality of region and subregions do not match.')
    if np.any(region < subregion_size):
        raise ValueError('Region is to small to contain subregion size.')
    if type(num_subregions) is int:
        num_subregions = [num_subregions] * dim_region
    elif dim_region != len(num_subregions):
        raise ValueError(
            'Dimensionality of region and number of subregions do not match.')

    range_rank = range(dim_region)
    max_valid_position = [
        region[rank] - subregion_size[rank] for rank in range_rank
    ]
    position_distance = [
        int(max_valid_position[rank] / (num_subregions[rank] - 1))
        for rank in range_rank
    ]
    subregions = [
        list(range(num_subregion)) for num_subregion in num_subregions
    ]

    subregions = list(itertools.product(*subregions))
    positions = np.multiply(subregions, position_distance).astype(int)

    return positions


class MRICropFirstVoxelCoordinates(Dataset):
    """Dataset of MRI crops.

    Its items are uniformly distributed crops of an 3D image and their first
    voxel coodinates.
    """

    def __init__(
        self,
        image: torch.Tensor,
        num_crops_per_direction: int,
        crop_size: Tuple,
    ):
        """Initialize image and crop informations."""
        self.image = image
        self.first_voxel_coordinates = distribute_regions_uniformely(
            region=list(self.image.shape),
            subregion_size=list(crop_size),
            num_subregions=num_crops_per_direction)
        self.crop_size = crop_size

    def __len__(self):
        """Provide length information."""
        return len(self.first_voxel_coordinates)

    def __getitem__(self, index):
        """Generate one crop of the image."""
        img = self.image

        start = self.first_voxel_coordinates[index]

        X = crop_image(image=img, start=start, size=self.crop_size)
        X = X[np.newaxis]  # C x H x W x D

        return {'coord': start, 'image': X.to(torch.float)}


def initialize_cropped_generator(
        image: torch.Tensor,
        batch_size: int,
        num_cpu_workers: int,
        num_crops_per_direction: int,
        crop_size: Tuple,
) -> DataLoader:
    """Initialize deployment generator."""
    deployment_dataset = MRICropFirstVoxelCoordinates(
        image=image,
        num_crops_per_direction=num_crops_per_direction,
        crop_size=crop_size)

    data_loader = DataLoader(
        dataset=deployment_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_cpu_workers,
    )

    return data_loader


def make_model_prediction(
        data_generator: DataLoader,
        model: nn.Module,
        prediction: torch.Tensor,
        normalize: bool = True,
) -> torch.Tensor:
    """
    Make a models prediction on a data sample.

    Parameters
    ----------
    data_generator : DataLoader
        Generates model inputs which are part of the sample.
    model : nn.Module
        Trained model.
    prediction : torch.Tensor
        Placeholder for prediction.

    Returns
    -------
    prediction : torch.Tensor
        Models prediction on data_generator.

    """
    if normalize:
        normalization_counter = torch.zeros(
            prediction.shape,
            requires_grad=False,
            dtype=int,
            device=prediction.device,
        )

    for sample_batched in data_generator:
        image_crop = sample_batched['image'].to(prediction.device)
        start_positions = sample_batched['coord']

        crop_prediction = model(image_crop)

        # cumulate prediction
        for i_batch, start in enumerate(start_positions):
            end = np.add(start, image_crop.shape[2:])
            prediction[:, :, start[0]:end[0], start[1]:end[1],
                       start[2]:end[2]] += crop_prediction[i_batch:i_batch + 1]

            if normalize:
                normalization_counter[:, :, start[0]:end[0], start[1]:end[1],
                                      start[2]:end[2]] += 1

    del start_positions, end, image_crop, crop_prediction

    if normalize:
        prediction = torch.div(prediction, normalization_counter)
        del normalization_counter

    return prediction
