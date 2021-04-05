#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/04/05 19:53

@author: Tei Koten
"""

import numpy as np
from scipy import interpolate


def fit_freq(array, originalFreq, alterFreq, kind='cubic'):
    """
    In order to cause 2 different sampling to be consistent.
    ------------------------------------------

    :param array: 1darray or ndarray of shape (n_timepoint,n_parameters)
    :param originalFreq: int or float
    :param alterFreq: int or float
    :param kind: string, default = 'cubic'
    :return: ndarray of shape (n_newtimepoint,n_parameters)

    ----------------Example------------------
    1000Hz --> 360Hz
    var = fit_freq(array,1000,360,kind='linear')
    var = fit_freq(array,1000,360,kind='cubic')
    var = fit_freq(array,1000,360,kind='quadratic')
    -----------------------------------------
    """
    length = len(array)
    alter_interval = originalFreq * length / alterFreq
    original_timeline = np.linspace(1, length, length)
    alter_timeline = np.linspace(1, length, num=round(alter_interval))
    intetp_func = interpolate.interp1d(original_timeline, array, kind=kind)
    res = intetp_func(alter_timeline)
    return res
