#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/12/24 19:28

@author: Tei Koten
"""

import numpy as np


def derivative5p(array, delta_t=1e-3):
    """
    5点微分法
    :param array: 1 dimensional array
    :param delta_t: ∆t = Hz
    :return: derivative array
    """
    res = list()
    array1 = (-25 * array[0] + 48 * array[1] - 36 * array[2] + 16 * array[3] - 3 * array[4]) / 12 * delta_t
    array2 = (-3 * array[0] - 10 * array[1] + 18 * array[2] - 6 * array[3] + array[4]) / 12 * delta_t
    res.append(array1)
    res.append(array2)
    for i in range(2, len(array) - 2):
        array_i = (array[i - 2] - 8 * array[i - 1] + 8 * array[i + 1] - array[i + 2]) / 12 * delta_t
        res.append(array_i)
    array_last2 = (-1 * array[0] + 6 * array[1] - 18 * array[2] + 10 * array[3] + 3 * array[4]) / 12 * delta_t
    array_last1 = (-3 * array[0] - 16 * array[1] + 36 * array[2] - 48 * array[3] + 25 * array[4]) / 12 * delta_t
    res.append(array_last2)
    res.append(array_last1)
    return np.array(res)
