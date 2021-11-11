#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/05/06 18:33

@author: Tei Koten
"""

import numpy as np
from scipy import signal
from scipy import interpolate


class SignalProcessing(object):
    """
    This class includes various common use tools about signal process for BMC.lab.
    This program is based on Biomechanics lab.
    ### Example ###

    """
    def fitFreq(self, array, originalFreq, alterFreq, kind='cubic'):
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
        alter_interval = alterFreq * length / originalFreq
        original_timeline = np.linspace(1, length, length)
        alter_timeline = np.linspace(1, length, num=round(alter_interval))
        intetp_func = interpolate.interp1d(original_timeline, array, kind=kind)
        res = intetp_func(alter_timeline)
        return res

    def filterSig(self,array, filterfreq, fzfreq=1000, order=2, btype='low'):
        m, n = array.shape
        filtedData = np.zeros((m, n))
        for i in range(n):
            param = np.float(2 * filterfreq / fzfreq)
            b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
            filtedData[:, i] = signal.filtfilt(b, a, array[:, i])  # data为要过滤的信号
        return filtedData




    
