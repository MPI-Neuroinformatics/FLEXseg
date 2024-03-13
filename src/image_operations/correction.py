#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 00:56:39 2024

@author: jsteiglechner

This module provides functionality to correct and unify images.
"""

import numpy as np


def clean_up_image(image: np.ndarray) -> np.ndarray:
    """
    Clean an image by removing infinite numbers and unnecessary dimensions.

    Parameters
    ----------
    image : np.ndarray
        an image.

    Returns
    -------
    np.ndarray
        cleaned image.

    """
    return np.nan_to_num(np.squeeze(image))
