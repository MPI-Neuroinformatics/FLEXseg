#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module provides functionality for handling ITK images.

Mainly, there are conversions to and from them.
"""

from typing import List, Tuple

import itk
import numpy as np

ITK_COORD_LPI: np.ndarray = np.array([-1, -1, 1], dtype=float)


def convert_affine_to_itk_params(
        affine: np.ndarray,
) -> Tuple[List, List, itk.Matrix]:
    """
    Get itk conform information from affine matrix.

    Parameters
    ----------
    affine : np.ndarray
        Affine matrix.

    Raises
    ------
    ValueError
        If affine spatial dimensionality is not 3.

    Returns
    -------
    Tuple[List, List, itk.Matrix]
        Affine space with itk conform values: spacing, origin, direction.

    """
    if affine.shape[0] != 4:
        raise ValueError(
            f'Spatial dimensionality is not 3, but {affine.shape[0] - 1}.'
        )

    # Extract affine informations
    affine_direction = affine[:3, :3]
    affine_spacing: np.ndarray = np.linalg.norm(affine_direction, axis=0)
    affine_origin = affine[:3, 3]

    # Convert affine informations
    affine_origin = affine_origin * ITK_COORD_LPI
    affine_direction = affine_direction / affine_spacing
    affine_direction = (affine_direction.T * ITK_COORD_LPI).T

    # Return itk conform values
    return (
        affine_spacing.tolist(),
        affine_origin.tolist(),
        itk.matrix_from_array(affine_direction),
    )


def convert_itk_params_to_affine(itk_img: itk.Image) -> np.ndarray:
    """
    Extract an affine matrix from itk image.

    Parameters
    ----------
    itk_img : itk.Image
        ITK image.

    Raises
    ------
    ValueError
        If image dimensionality does not match.

    Returns
    -------
    affine : np.ndarray
        Affine matrix.

    """
    if itk_img.GetImageDimension() != 3:
        raise ValueError(
            f'Image dimensionality isnot 3, but {itk_img.GetImageDimension()}.'
        )

    # Extract image information
    affine_origin = itk_img.GetOrigin()
    affine_spacing = itk_img.GetSpacing()
    affine_direction = itk.array_from_matrix(itk_img.GetDirection())

    # Convert affine information
    affine_origin = affine_origin * ITK_COORD_LPI
    affine_direction = affine_direction * affine_spacing
    affine_direction = (affine_direction.T * ITK_COORD_LPI).T

    # Conclude affine information
    affine = np.eye(4, dtype=float)
    affine[:3, :3] = affine_direction
    affine[:3, 3] = affine_origin

    return affine


def convert_nifti_array_to_itk(
        img: np.ndarray,
        affine: np.ndarray,
        voxel_type: itk.itkCType = itk.F,
) -> itk.Image:
    """
    Convert a physical 3D-image to an ITK Image.

    Parameters
    ----------
    img : np.ndarray
        Input image effectivly in F ordering.
    affine : np.ndarray
        Affine matrix defining the physical cartesian space of image grid.
    voxel_type : itk.itkCType, optional
        Data type of voxels. The default is itk.F.

    Raises
    ------
    ValueError
        If image dimensionality is not 3.
        If affine spatial dimensionality is not 3.

    Returns
    -------
    itk_img :  itk.Image
        Converted ITK image.

    """
    if len(img.shape) != 3:
        raise ValueError(
            f'Image dimensionality is not 3, but {len(img.shape)}.')

    if img.flags['C_CONTIGUOUS']:
        # Is set to F-Odering so that shapes of np and ITK image match.
        # Note, that NIfTI1 images are effectivly in F-Ordering.
        img = np.asfortranarray(img)

    affine_spacing, affine_origin, affine_direction = \
        convert_affine_to_itk_params(affine=affine)

    # Convert to ITK, data is copied
    itk_img = itk.image_from_array(img)
    itk_img = itk_img.astype(voxel_type)

    # Set image informations
    itk_img.SetOrigin(affine_origin)
    itk_img.SetSpacing(affine_spacing)
    itk_img.SetDirection(affine_direction)

    return itk_img


def convert_itk_img_to_nifti_array(
        itk_img: itk.Image,
        dtype: np.dtype = float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert an ITK image to a Numpy array with a defined physical space.

    Parameters
    ----------
    itk_img : itk.Image
        Image in ITK format.
    dtype : np.dtype, optional
        Voxel dtype. The default is float.

    Returns
    -------
    img : TYPE
        Image arrage in F-ordering.
    affine : TYPE
        Corresponding affine matrix.

    """
    affine = convert_itk_params_to_affine(itk_img)

    # Convert to array, data is copied.
    # This will always be a C ordered array.
    # Has to be transposed to match NIfTI1 convention of F-Ordering.
    img = itk.array_from_image(itk_img).T
    img = img.astype(dtype)

    return img, affine
