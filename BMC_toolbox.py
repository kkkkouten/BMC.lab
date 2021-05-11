#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/05/06 18:33

@author: Tei Koten
"""

import numpy as np
from scipy import signal
from scipy import interpolate


class BMC_lab(object):
    """
    This class includes various common use tools for BMC.lab.
    This program is based on Biomechanics lab.
    parameters:
    * data:
    *
    *
    *
    *
    ### Example ###
    """
    def __init__(self, data):
        self.dat = data

    def CoP(self, Mx, My, Fz):
        """
        :param Mx: 1d array
        :param My: 1d array
        :param Fz: 1d array
        :return:
        """
        COP_x = My / Fz
        COP_y = -Mx / Fz
        return np.array([COP_x, COP_y]).T

    def free_moment(self, CoP, Fx, Fy, Mz):
        """
        :param COP: 2d array
        :param Fx: 1d array
        :param Fy: 1d array
        :param Mz: 1d array
        :return:  FM
        """
        COP_x, COP_y = CoP[:, 0], COP[:, 1]
        FM = Mz - Fy * COP_x + Fx * COP_y
        return FM

    def derivative5p(self, array, delta_t=1e-3):
        """
        5点微分法
        :param array: 1 dimensional array
        :param delta_t: ∆t = 1/freq
        :return: derivative array
        """
        res = list()
        array1 = (-25 * array[0] + 48 * array[1] - 36 * array[2] +
                  16 * array[3] - 3 * array[4]) / (12 * delta_t)
        array2 = (-3 * array[0] - 10 * array[1] + 18 * array[2] -
                  6 * array[3] + array[4]) / (12 * delta_t)
        res.append(array1)
        res.append(array2)
        for i in range(2, len(array) - 2):
            array_i = (array[i - 2] - 8 * array[i - 1] +
                       8 * array[i + 1] - array[i + 2]) / (12 * delta_t)
            res.append(array_i)
        array_last2 = (-1 * array[0] + 6 * array[1] - 18 * array[2] +
                       10 * array[3] + 3 * array[4]) / (12 * delta_t)
        array_last1 = (-3 * array[0] - 16 * array[1] + 36 * array[2] -
                       48 * array[3] + 25 * array[4]) / (12 * delta_t)
        res.append(array_last2)
        res.append(array_last1)
        return np.array(res)

    def fit_freq(self, array, originalFreq, alterFreq, kind='cubic'):
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

    def filter_sig(self, array, cut_off_freq, fzfreq=1000, order=2, btype='lowpass'):
        """
        :param array: 1d array
        :param cut_off_freq: int
        :param fzfreq: int, default = 1000Hz
        :param order: int, default = 2
        :param btype: string, btype,{'lowpass', 'highpass', 'bandpass', 'bandstop'}
        :return:
        """
        param = np.float(2 * cut_off_freq / fzfreq)
        b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, array)  # data为要过滤的信号
        return filtedData
