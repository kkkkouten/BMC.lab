#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/09/30 16:20

@author: Tei Koten
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal


class preprocessing(object):

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

    def findSync(self, dat):
        nums = np.where(dat.Fz > 10)[0]
        return [nums[0], nums[-1]]

    def filterSig(self,array, filterfreq, fzfreq=1000, order=2, btype='low'):
        m, n = array.shape
        filtedData = np.zeros((m, n))
        for i in range(n):
            param = np.float(2 * filterfreq / fzfreq)
            b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
            filtedData[:, i] = signal.filtfilt(b, a, array[:, i])  # data为要过滤的信号
        return filtedData

    def batchProcess(self, dat, filterfreq, fzfreq=2000):
        res = dat.loc[:, ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']].values
        ans = pd.DataFrame()
        m, n = res.shape
        for j in range(n):
            temp = self.filterSig(res[:, j], filterfreq, fzfreq=fzfreq)  # filter
            tempFit = pd.DataFrame(self.fitFreq(temp, originalFreq=fzfreq, alterFreq=100))
            res[:, j] = temp
            ans = pd.concat((ans, tempFit), axis=1)
        res = pd.DataFrame(res)
        res.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        ans.columns = res.columns
        return res, ans

    def getCOP(self, fp):
        COP_x = fp.My / fp.Fz
        COP_y = -1 * fp.Mx / fp.Fz
        COP_z = [0] * len(fp)
        return np.array([COP_x, COP_y, COP_z]).T

    def derivative5p(self, array, order=1, delta_t=1e-3):
        """
        5点微分法
        :param array: 1 dimensional array
        :param delta_t: ∆t = 1/freq
        :return: derivative array
        """
        l = len(array)
        res = list()
        if order == 1:
            array1 = (-25 * array[0] + 48 * array[1] - 36 * array[2] +
                      16 * array[3] - 3 * array[4]) / (12 * delta_t)
            array2 = (-3 * array[0] - 10 * array[1] + 18 * array[2] -
                      6 * array[3] + array[4]) / (12 * delta_t)
            res.append(array1)
            res.append(array2)
            for i in range(2, l - 2):
                array_i = (array[i - 2] - 8 * array[i - 1] +
                           8 * array[i + 1] - array[i + 2]) / (12 * delta_t)
                res.append(array_i)
            array_last2 = (-1 * array[0] + 6 * array[1] - 18 * array[2] +
                           10 * array[3] + 3 * array[4]) / (12 * delta_t)
            array_last1 = (-3 * array[0] - 16 * array[1] + 36 * array[2] -
                           48 * array[3] + 25 * array[4]) / (12 * delta_t)
            res.append(array_last2)
            res.append(array_last1)
        elif order == 2:
            array1 = (2 * array[0] - 5 * array[1] + 4 * array[2] - array[3]) / (delta_t ** 2)
            array2 = (array[0] - 2 * array[1] + array[2]) / (delta_t ** 2)
            res.append(array1)
            res.append(array2)
            for i in range(2, l - 2):
                array_i = (-array[i - 2] + 16 * array[i - 1] - 30 * array[i]
                           + 16 * array[i + 1] - array[i + 2]) / (12 * delta_t ** 2)
                res.append(array_i)
            array_last2 = (array[l - 3] - 2 * array[l - 2] + array[l - 1]) / (delta_t ** 2)
            array_last1 = (array[l - 4] - 4 * array[l - 3] + 5 * array[l - 2] - 2 * array[l - 1]) / (delta_t ** 2)
            res.append(array_last2)
            res.append(array_last1)
        else:
            raise ValueError("n")
        return np.array(res)


if __name__ == "__main__":
    # labchart 2000Hz
    dat = pd.read_table("/Users/kkkkouten/BMC.lab/NMB/20201127/processed/1.txt", sep=",")
    bp = basicProcess()
    dat, ans = bp.batchProcess(dat, 40, 2000)
    ind = bp.findSync(ans)
    res = ans.iloc[ind[0]:ind[1], :]
    res.index = [i for i in range(len(res))]

    x = [i / max(res.index) * 100 for i in range(len(res.index))]
    plt.plot(x, res.Fz)
    plt.xticks([i * 10 for i in range(11)],
               ["{}%".format(i * 10) for i in range(11)])
    plt.show()
