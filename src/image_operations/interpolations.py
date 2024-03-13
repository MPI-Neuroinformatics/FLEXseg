#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

Module to perform interpolations for domain data points.

Possible future interpolations:
* https://en.wikipedia.org/wiki/Natural_neighbor_interpolation
* https://en.wikipedia.org/wiki/Multivariate_interpolation
* https://en.wikipedia.org/wiki/Triangulated_irregular_network
* https://en.wikipedia.org/wiki/Inverse_distance_weighting
* majority resampling
"""

import numpy as np


def get_domain_values_at_positions(
    domain_values: np.ndarray,
    set_of_kernel_points: np.ndarray,
) -> np.ndarray:
    r"""Get domain value for each raster position.

    Set to zero if position is not in domain raster.
    """
    set_of_kernel_points_moved = np.moveaxis(set_of_kernel_points, -2, -1)

    are_positions_in_img = (
        np.all(set_of_kernel_points_moved >= 0, axis=-1)
        & np.all(set_of_kernel_points_moved < domain_values.shape, axis=-1))
    positions_in_img = set_of_kernel_points_moved[are_positions_in_img]
    codomain_values = np.zeros(
        set_of_kernel_points_moved.shape[:-1],
        dtype=domain_values.dtype,
    )
    codomain_values[are_positions_in_img] = domain_values[
        positions_in_img[:, 0],
        positions_in_img[:, 1],
        positions_in_img[:, 2],
    ]

    return codomain_values


def linear_distance_weighting_of_points(
    set_of_points: np.ndarray,
    set_of_kernel_points: np.ndarray,
) -> np.ndarray:
    r"""Get linear weights for each raster point to a containing point."""
    out = np.subtract(set_of_kernel_points,
                      set_of_points[..., np.newaxis]).astype(np.float16)
    out = np.abs(out)
    out = np.subtract(1., out)

    return np.prod(out, axis=-2)


def linear_interpolation(weights: np.ndarray,
                         values: np.ndarray) -> np.ndarray:
    """Interpolate linerally weight-value pairs for certain points."""
    return np.divide(np.sum(np.multiply(weights, values), axis=-1),
                     np.sum(weights, axis=-1))


def nearest_neightbour(
    values: np.ndarray,
    set_of_points: np.ndarray,
) -> np.ndarray:
    """Find the nearest neighbour values of points in a raster."""
    return get_domain_values_at_positions(
        domain_values=values, set_of_kernel_points=np.around(set_of_points))


def weighted_majority_resampling(
    weights: np.ndarray,
    values: np.ndarray,
    unique_values: np.ndarray,
) -> np.ndarray:
    """Resample values with a majority of weights."""
    if not np.issubdtype(values.dtype, np.integer):
        values = values.astype(np.int16)
        unique_values = unique_values.astype(np.int16)

    num_values = len(np.unique(values))
    values = np.searchsorted(unique_values, values).astype(unique_values.dtype)

    one_hot_encoded_values = np.eye(num_values, dtype=values.dtype)[values]
    weights = one_hot_encoded_values * weights[..., np.newaxis]

    del one_hot_encoded_values, values
    weights = np.sum(weights, axis=-2)
    weights = np.argmax(weights, axis=-1)

    return unique_values[weights]
