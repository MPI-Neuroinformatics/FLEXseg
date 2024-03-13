#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module provides functionality for geometric transformations.
"""

from typing import Tuple

import numpy as np

from image_operations.interpolations import (
    get_domain_values_at_positions,
    linear_distance_weighting_of_points,
    weighted_majority_resampling,
)


def get_relative_kernel_coordinates(kernel_size: int) -> np.ndarray:
    """Build the relative kernel coordinates from origin."""
    if kernel_size == 1:
        return np.array([[0, 0, 0]], dtype=np.int8).T
    elif kernel_size == 2:
        return np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1],
                         [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],
                        dtype=np.int8).T  # yapf: disable
    elif kernel_size == 3:
        return np.array(
            [[-1, -1, -1], [-1, -1, 0], [-1, -1, 1], [-1, 0, -1], [-1, 0, 0],
             [-1, 0, 1], [-1, 1, -1], [-1, 1, 0], [-1, 1, 1],
             [0, -1, -1], [0, -1, 0], [0, -1, 1], [0, 0, -1], [0, 0, 0],
             [0, 0, 1], [0, 1, -1], [0, 1, 0], [0, 1, 1],
             [1, -1, -1], [1, -1, 0], [1, -1, 1], [1, 0, -1], [1, 0, 0],
             [1, 0, 1], [1, 1, -1], [1, 1, 0], [1, 1, 1]],
            dtype=np.int8).T  # yapf: disable
    else:
        raise ValueError(f"Kernel size {kernel_size} isn't implemented yet.")


def kernel_coordinates(set_of_points: np.ndarray,
                       kernel_size: int) -> np.ndarray:
    """Build the associated points in a kernel."""
    floored_set_of_points = np.floor(set_of_points).astype(np.int16)
    relative_kernel_coordinates = get_relative_kernel_coordinates(
        kernel_size=kernel_size)

    absolute_kernel_coordinates = np.add(
        floored_set_of_points[..., np.newaxis],
        relative_kernel_coordinates).astype(np.int16)

    if np.max(absolute_kernel_coordinates) < 128:
        absolute_kernel_coordinates = absolute_kernel_coordinates.astype(
            np.int8)
    else:
        absolute_kernel_coordinates = absolute_kernel_coordinates.astype(
            np.int16)

    return absolute_kernel_coordinates


def get_raster_augmented_coordinates(
        raster_shape: Tuple[int, int, int]) -> np.ndarray:
    """Extract all augmented indices in a raster."""
    raster_coordinates = np.ones((4, ) + raster_shape, dtype=np.int16)
    raster_coordinates[:3] = np.indices(raster_shape)

    return raster_coordinates


def affine_coordinates_transform(affine_map: np.ndarray,
                                 set_of_points: np.ndarray) -> np.ndarray:
    """Apply an affine map on a set of augmented points."""
    x_set = np.moveaxis(set_of_points, 0, -2)
    y_set = np.matmul(affine_map, x_set, dtype=np.float16)
    y_set = np.moveaxis(y_set, -2, -1)

    return y_set[..., :3]


def affine_image_transform(
        image: np.ndarray,
        affine_transform_matrix: np.ndarray,
        target_shape: np.ndarray,
        interpolation: str,
        interpolation_kernel_size: int = 2,
) -> np.ndarray:
    """
    Relocate pixels requiring intensity interpolation.

    Approximate the value of moved pixels.

    """
    codomain_raster_coordinates = get_raster_augmented_coordinates(
        raster_shape=target_shape)
    domain_coordinates = affine_coordinates_transform(
        affine_map=affine_transform_matrix,
        set_of_points=codomain_raster_coordinates)
    del codomain_raster_coordinates
    domain_kernel_coordinates = kernel_coordinates(
        set_of_points=domain_coordinates,
        kernel_size=interpolation_kernel_size)
    domain_values = get_domain_values_at_positions(
        domain_values=image, set_of_kernel_points=domain_kernel_coordinates)
    domain_weights = linear_distance_weighting_of_points(
        set_of_points=domain_coordinates,
        set_of_kernel_points=domain_kernel_coordinates)
    del domain_kernel_coordinates

    if interpolation == 'majority_voted':
        del domain_coordinates
        return weighted_majority_resampling(
            weights=domain_weights.astype(np.float16),
            values=domain_values,
            unique_values=np.unique(image),)
    else:
        raise ValueError(
            f'Interpolation method {interpolation} is not available.')
