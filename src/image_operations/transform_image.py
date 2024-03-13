#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module provides the functionality to perform a affine transformation."""

from typing import Tuple

import numpy as np

from image_operations.geometric_transformations import affine_image_transform


def get_domain_limit(domain_shape: Tuple[int, int, int]) -> np.array:
    """Get the limit of the domain containing the domain shape."""
    x_upper, y_upper, z_upper = domain_shape
    x_upper, y_upper, z_upper = x_upper - 1, y_upper - 1, z_upper - 1
    augmented_domain_limit = np.array([
        [0, 0, 0, 1],
        [x_upper, 0, 0, 1],
        [0, y_upper, 0, 1],
        [0, 0, z_upper, 1],
        [x_upper, y_upper, 0, 1],
        [x_upper, 0, z_upper, 1],
        [0, y_upper, z_upper, 1],
        [x_upper, y_upper, z_upper, 1],
    ], dtype=int)
    return augmented_domain_limit


def get_image_limit_in_codomain(
        domain_shape: Tuple[int, int, int],
        affine_transform_matrix: np.ndarray) -> np.ndarray:
    """Get the limit of the codomain by transforming the domain shape."""
    limit_in_domain = get_domain_limit(domain_shape).T
    limit_in_codomain = np.matmul(affine_transform_matrix, limit_in_domain)[:3]

    coordinate_limit_in_codomain = list(
        zip(limit_in_codomain.min(axis=-1), limit_in_codomain.max(axis=-1)))

    return coordinate_limit_in_codomain


def add_translation_of_limit_to_affine(affine_matrix, limit):
    """Add the origin of a limit to a affine while adjusting the limit."""
    affine_matrix_tmp = np.copy(affine_matrix)
    (x_lower, x_upper), (y_lower, y_upper), (z_lower, z_upper) = limit
    affine_matrix_tmp[:3, 3] = np.dot(affine_matrix_tmp[:3, :3],
                                      [x_lower, y_lower, z_lower])

    limit = [(0, x_upper - x_lower), (0, y_upper - y_lower),
             (0, z_upper - z_lower)]

    return affine_matrix_tmp, limit


def expand_image_limit(image_limit: np) -> Tuple[int, int, int]:
    """Expand the limit by 1 to surely contain the whole image in a raster."""
    _, upper_limit = np.array(list(zip(*image_limit)))
    if np.any(upper_limit < 0):
        raise ValueError(
            'The intersection of planned image and kernel of data is empty.'
        )

    expanded_upper_limit = np.ceil(upper_limit) + 1

    return tuple(expanded_upper_limit.astype(int))


def complete_target_affine_space(
    affine_matrix: np.ndarray,
    target_affine_matrix: np.ndarray,
    image_shape: Tuple[int, int, int],
    target_shape: Tuple[int, int, int] = None,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    """
    Complete affine space by adding a translation and space size.

    Parameters
    ----------
    affine_matrix : np.ndarray
        Domain affine.
    target_affine_matrix : np.ndarray
        Codomain affine.
    image_shape : Tuple[int, int, int]
        Domain image shape.
    target_shape : Tuple[int, int, int], optional
        Codomain image shape. The default is None.

    Raises
    ------
    ValueError
        If codomain image shape has wrong dimensionality.
        If codomain image shape is given without origin.

    Returns
    -------
    target_affine_matrix : np.ndarray
        Affine of codomain space.
    target_shape : Tuple[int, int, int]
        Size of codomain space.

    """
    if target_shape is not None and not (len(target_shape) == 3):
        raise ValueError(
            'The target shape specifies a wrong dimensionality')
    if target_shape is not None and target_affine_matrix.shape != (4, 4):
        raise ValueError(
            'A target shape needs a given translation but it is not given.')

    # Case sensitive if augmented target affine comes with a given translation
    if target_affine_matrix.shape == (3, 3):
        translation_is_missing = True
        target_affine_matrix = np.append(target_affine_matrix, [[0, 0, 0]], 0)
        target_affine_matrix = np.append(target_affine_matrix,
                                         [[0], [0], [0], [1]], 1)
    else:
        translation_is_missing = False

    inverse_affine_transform_matrix = np.dot(
        np.linalg.inv(target_affine_matrix), affine_matrix)
    image_limit_in_codomain_space = get_image_limit_in_codomain(
        domain_shape=image_shape,
        affine_transform_matrix=inverse_affine_transform_matrix)
    if translation_is_missing:
        target_affine_matrix, image_limit_in_codomain_space = \
            add_translation_of_limit_to_affine(
                affine_matrix=target_affine_matrix,
                limit=image_limit_in_codomain_space,)

    if target_shape is None:
        target_shape = expand_image_limit(
            image_limit=image_limit_in_codomain_space)

    return target_affine_matrix, target_shape


def transform_image(
        image: np.ndarray,
        affine_matrix: np.ndarray,
        target_affine_matrix: np.ndarray,
        target_shape: Tuple[int, int, int] = None,
        interpolation: str = 'linear',
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform an affine transformation to an image.

    Parameters
    ----------
    image : np.ndarray
        Domain image.
    affine_matrix : np.ndarray
        Domain affine space.
    target_affine_matrix : np.ndarray
        Codomain affine space.
    target_shape : Tuple[int, int, int], optional
        Codomain image shape. The default is None.
    interpolation : str, optional
        Interpolation method. The default is 'linear'.

    Raises
    ------
    ValueError
        If affine spaces contain infinite numbers.
        If image contains infinite numbers.

    Returns
    -------
    np.ndarray
        Codomain image.
    np.ndarray
        Codomain affine.

    """
    if np.any(np.isnan(affine_matrix)) or np.any(
            np.isnan(target_affine_matrix)):
        raise ValueError('Affine Matrices contain infinite numbers.')
    if np.any(np.isnan(image)):
        raise ValueError('Image contains infinite numbers.')

    # Check if transform is necessary
    if target_affine_matrix.shape != (4, 4):
        affine_to_compare = affine_matrix[:3, :3]
    else:
        affine_to_compare = affine_matrix
    if np.allclose(affine_to_compare,
                   target_affine_matrix) and image.shape is target_shape:
        return image, affine_matrix

    target_affine_matrix, target_shape = complete_target_affine_space(
        affine_matrix=affine_matrix,
        target_affine_matrix=target_affine_matrix,
        image_shape=image.shape,
        target_shape=target_shape,
    )

    affine_transform_matrix = np.dot(np.linalg.inv(affine_matrix),
                                     target_affine_matrix)

    transformed_image = affine_image_transform(
        image=image,
        affine_transform_matrix=affine_transform_matrix,
        target_shape=target_shape,
        interpolation=interpolation)

    return transformed_image, target_affine_matrix
