#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 28 09:31:25 2024

@author: jsteiglechner
"""

import numpy as np


def quantile_clipping(
        feature: np.array,
        lower_quantile: float = 0.01,
        upper_quantile: float = 0.99,
) -> np.ndarray:
    lower_limit = np.quantile(feature, lower_quantile)
    upper_limit = np.quantile(feature, upper_quantile)

    feature = np.where(
        feature < lower_limit,
        np.min(feature[feature >= lower_limit]),
        feature,
    )
    feature = np.where(
        feature > upper_limit,
        np.max(feature[feature <= upper_limit]),
        feature,
    )

    return feature
