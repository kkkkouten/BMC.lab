#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/09/30 22:59

@author: Tei Koten
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy import signal

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/preprocessing.py")
from preprocessing import preprocessing


class Kinematics(object):
    def __init__(self, dat):
        self.dat = dat
        self.dt = 1 / 360

    def versors(self, m1, m2, m3):
        v1 = m2 - m1  # first axis
        v2 = np.cross(v1, m3 - m1)  # second axis
        v3 = np.cross(v1, v2)  # third axis
        # v1 = np.cross(v3, v2)
        v1norm = np.linalg.norm(v1)
        v2norm = np.linalg.norm(v2)
        v3norm = np.linalg.norm(v3)
        # Vector normalization
        e1 = v1 / v1norm
        e2 = v2 / v2norm
        e3 = v3 / v3norm
        return e1, e2, e3

    def localFrame(self, m1, m2, m3):
        n = len(m1)
        base = list()
        for i in range(n):
            e1, e2, e3 = self.versors(m1[i], m2[i], m3[i])
            basis = np.vstack((e1, e2, e3))
            base.append(basis)
        return np.array(base)

    def getbase(self, m1, m2, m3):
        n = len(m1)
        E1 = [np.array([0, 0, 0])]
        E2 = [np.array([0, 0, 0])]
        E3 = [np.array([0, 0, 0])]
        for i in range(n):
            e1, e2, e3 = self.versors(m1[i], m2[i], m3[i])
            E1.append(e1)
            E2.append(e2)
            E3.append(e3)
        return np.array(E1), np.array(E2), np.array(E3)

    def testLocalFrame(self, m1, m2, m3):
        n = len(m1)
        orths = list()
        norms = list()
        for i in range(n):
            e1, e2, e3 = self.versors(m1[i], m2[i], m3[i])
            orth = [np.linalg.norm(np.cross(e1, e2)),
                    np.linalg.norm(np.cross(e1, e3)),
                    np.linalg.norm(np.cross(e2, e3))]
            norm = [np.linalg.norm(e1),
                    np.linalg.norm(e2),
                    np.linalg.norm(e3)]
            orths.append(orth)
            norms.append(norm)
        return np.array(orths), np.array(norms)

    def getOmega(self, e, dt=1 / 360):
        e1 = e[:, 0, :]
        e2 = e[:, 1, :]
        e3 = e[:, 2, :]
        de1dt = self.derivative5p(e1, order=1,delta_t=dt)
        de2dt = self.derivative5p(e2, order=1,delta_t=dt)
        de3dt = self.derivative5p(e3, order=1,delta_t=dt)
        omega1 = np.sum(de2dt * e3[0:-1, :], axis=1).reshape(-1, 1)
        omega2 = np.sum(de3dt * e1[0:-1, :], axis=1).reshape(-1, 1)
        omega3 = np.sum(de1dt * e2[0:-1, :], axis=1).reshape(-1, 1)
        omega = np.hstack((omega1, omega2, omega3))
        return omega

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

    def filterSig(self, array, filterfreq, fzfreq=1000, order=2, btype='low'):
        param = np.float(2 * filterfreq / fzfreq)
        b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
        filtedData = signal.filtfilt(b, a, array)  # data为要过滤的信号
        return filtedData

if __name__ == "__main__":
    #
    #
    path = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    dat = pd.read_csv(path, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    #
    for i, j in enumerate(dat.columns):
        print(i, j)

    # 建立坐标系
    kinema = Kinematics(dat)

    # Ankle coordinate system
    lAnk = dat.loc[:, "NMB:R_LANK"].values
    mAnk = dat.loc[:, "NMB:R_MANK"].values
    iAnk = 0.5 * (lAnk + mAnk)
    m1 = iAnk
    m2 = mAnk
    m3 = dat.loc[:, "NMB:R_FBK"].values

    ankleBase = kinema.localFrame(m1, m2, m3)
    ankleOrigin = iAnk
    orths, norms = kinema.testLocalFrame(m1, m2, m3)

    # TibiaFibula coordinate system
    lTep = dat.loc[:, "NMB:R_LTEP"].values
    mTep = dat.loc[:, "NMB:R_MTEP"].values
    iTep = 0.5 * (lTep + mTep)
    m1 = iTep
    m2 = mTep
    m3 = dat.loc[:, "NMB:R_TIB"].values
    #
    kneeBase = kinema.localFrame(m1, m2, m3)
    dt = 1/360
    e1,e2,e3 = kinema.getbase(m1,m2,m3)
    de1dt = kinema.derivative5p(e1, order=1, delta_t=dt)
    de2dt = kinema.derivative5p(e2, order=1, delta_t=dt)
    de3dt = kinema.derivative5p(e3, order=1, delta_t=dt)
    omega1 = np.sum(de2dt * e3, axis=1).reshape(-1,1)
    omega2 = np.sum(de3dt * e1, axis=1).reshape(-1, 1)
    omega3 = np.sum(de1dt * e2, axis=1).reshape(-1, 1)
    omega = np.hstack((omega1, omega2, omega3))
    omega = kinema.getOmega(kneeBase)


    de2dt * e3


    plt.plot(omega[1617:1728])
    plt.show()

    kneeOrigin = iAnk
    # orths,norms = kinema.testLocalFrame(m1,m2,m3)
    #
    # # Femoral coordinate system
    # lFep = dat.loc[:, "NMB:R_LFEP"].values
    # mFep = dat.loc[:, "NMB:R_MFEP"].values
    # iFep = 0.5 * (lFep + mFep)
    # m1 = iFep
    # m2 = mFep
    # m3 = dat.loc[:, "NMB:R_ATH"].values
    # thighBase = kinema.localFrame(m1, m2, m3)
    # e1, e2, e3 = kinema.getbase(m1, m2, m3)
    # thighOrigin = iFep
    # # orths,norms = kinema.testLocalFrame(m1,m2,m3)
    #
    # # Femoral coordinate system
