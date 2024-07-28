#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner
"""

from typing import Tuple
import warnings

import itk
import numpy as np

from image_operations.itk_operations import (
    convert_nifti_array_to_itk,
    convert_affine_to_itk_params,
    convert_itk_img_to_nifti_array,
)
from image_operations.transform_image import (
    complete_target_affine_space
)


def affine_transform_to_reference(
        img_afp: str,
        reference_afp: str,
        destination_afp: str = None,
        interpolation: str = 'linear',
) -> itk.Image:
    """
    Perform an affine transformation to a domain image to match a reference.

    Parameters
    ----------
    img_afp : str
        Path to domain image.
    reference_afp : str
        Path to codomain defining image.
    destination_afp : str, optional
        Destination path to save transformed image. The default is None.
    interpolation : str, optional
        Interpolation method. The default is 'linear'.

    Raises
    ------
    ValueError
        If wrong interpolation method is specified.

    Returns
    -------
    resampled : itk.Image
        Transformed image.

    """
    mask = itk.imread(img_afp, itk.F)

    if interpolation == 'linear':
        interpolator = itk.LinearInterpolateImageFunction.New(mask)
    elif interpolation == 'nearest':
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(mask)
    else:
        raise ValueError('Wrong interpolation')

    TransformType = itk.AffineTransform[itk.D, 3]
    transform = TransformType.New()

    resampled = itk.resample_image_filter(mask,
                                          transform=transform,
                                          interpolator=interpolator,
                                          use_reference_image=True,
                                          reference_image=itk.imread(
                                              reference_afp, itk.F))

    if destination_afp:
        itk.imwrite(resampled, destination_afp)

    return resampled


def affine_transform_image(
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
    if not target_affine_matrix.shape == (4, 4):
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

    if image.dtype == np.dtype(np.float64):
        voxel_type = itk.D
    else:
        voxel_type = itk.F

    itk_image = convert_nifti_array_to_itk(img=image,
                                           affine=affine_matrix,
                                           voxel_type=voxel_type)

    if interpolation == 'linear':
        interpolator = itk.LinearInterpolateImageFunction.New(itk_image)
    elif interpolation == 'nearest':
        interpolator = itk.NearestNeighborInterpolateImageFunction.New(
            itk_image)

    dim = itk_image.GetImageDimension()
    transform = itk.AffineTransform[itk.D, dim].New()

    target_affine_information = convert_affine_to_itk_params(
        target_affine_matrix)
    target_size = itk.Size[len(target_shape)](np.array(target_shape,
                                                       dtype=int).tolist())

    resampled = itk.resample_image_filter(
        itk_image,
        interpolator=interpolator,
        transform=transform,
        default_pixel_value=0,
        output_spacing=target_affine_information[0],
        output_origin=target_affine_information[1],
        output_direction=target_affine_information[2],
        size=target_size,
    )

    image_resampled, affine_resampled = convert_itk_img_to_nifti_array(
        resampled)

    if image_resampled.dtype != image.dtype:
        if np.issubdtype(image.dtype, np.integer):
            image_resampled = np.rint(image_resampled, dtype=image.dtype)
        elif np.issubdtype(image.dtype, np.floating):
            image_resampled = image_resampled.astype(image.dtype)
        else:
            warnings.warn(f"Unsupported dtype {image.dtype}", Warning)

    return image_resampled, affine_resampled
