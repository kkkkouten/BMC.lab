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
from scipy import signal

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/preprocessing.py")
from preprocessing import preprocessing


class CoordinateSystem(object):

    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.dt = 1 / 360
        self.preprocessing = preprocessing()

    def main(self):
        self.versors()
        self.local_coordinate_system()
        self.angular_velocity()

    def versors(self):
        v1 = self.p2 - self.p1  # first axis
        v2 = np.cross(v1, self.p3 - self.p1)  # second axis
        v3 = np.cross(v1, v2)  # third axis
        v1 = np.cross(v3, v2)
        # v1norm = np.linalg.norm(v1)
        # v2norm = np.linalg.norm(v2)
        # v3norm = np.linalg.norm(v3)
        # # Vector normalization
        # self.e1 = v1 / v1norm
        # self.e2 = v2 / v2norm
        # self.e3 = v3 / v3norm
        v1norm = np.linalg.norm(v1, axis=1)
        v2norm = np.linalg.norm(v2, axis=1)
        v3norm = np.linalg.norm(v3, axis=1)
        e1 = np.zeros((v1.shape))
        e2 = np.zeros((v2.shape))
        e3 = np.zeros((v3.shape))
        for i in range(3):
            e1[:, i] = v1[:, i] * (1 / v1norm)
            e2[:, i] = v2[:, i] * (1 / v2norm)
            e3[:, i] = v3[:, i] * (1 / v3norm)
        self.e1 = e1
        self.e2 = e2
        self.e3 = e3

    def local_coordinate_system(self):
        e = list()
        for i in range(len(self.p1)):
            e.append(np.vstack((self.e1[i], self.e2[i], self.e3[i])).T)
        self.e = np.array(e)

    def angular_velocity(self):
        e_O = np.array([-1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)
        e_B = self.e
        A_OB = np.zeros((len(self.e), 3, 3))
        A_OB_dot = np.zeros((len(self.e), 3, 3))
        for i in range(len(self.e)):
            A_OB[i] = e_O @ e_B[i].T
        for i in range(3):
            for j in range(3):
                A_OB_dot[:, i, j] = self.preprocessing.derivative5p(A_OB[:, i, j], delta_t=1 / 360)
        tilde_omega = np.zeros((len(self.e), 3, 3))
        for i in range(len(self.e)):
            tilde_omega[i] = A_OB[i].T @ A_OB_dot[i]
        self.angular_velocity = np.vstack((tilde_omega[:, 2, 1], tilde_omega[:, 0, 2], tilde_omega[:, 1, 0])).T

    def test_local_coordinate_system(self):
        orths = list()
        norms = list()
        for i in range(len(self.p1)):
            orth = [np.linalg.norm(np.cross(self.e1[i], self.e2[i])),
                    np.linalg.norm(np.cross(self.e1[i], self.e3[i])),
                    np.linalg.norm(np.cross(self.e2[i], self.e3[i]))]
            norm = [np.linalg.norm(self.e1[i]), np.linalg.norm(self.e2[i]), np.linalg.norm(self.e3[i])]
            orths.append(orth)
            norms.append(norm)
        return np.array(orths), np.array(norms)


def derivative5p(array, order=1, delta_t=1e-3):
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


def filterSig(array, filterfreq, fzfreq=1000, order=2, btype='low'):
    m, n = array.shape
    filtedData = np.zeros((m, n))
    for i in range(n):
        param = np.float(2 * filterfreq / fzfreq)
        b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
        filtedData[:, i] = signal.filtfilt(b, a, array[:, i])  # data为要过滤的信号
    return filtedData


def filterSig(array, filterfreq, fzfreq=1000, order=2, btype='low'):
    m, n = array.shape
    filtedData = np.zeros((m, n))
    for i in range(n):
        param = np.float(2 * filterfreq / fzfreq)
        b, a = signal.butter(order, param, btype)  # 配置滤波器 8 表示滤波器的阶数
        filtedData[:, i] = signal.filtfilt(b, a, array[:, i])  # data为要过滤的信号
    return filtedData


if __name__ == "__main__":
    #
    #
    path = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    dat = pd.read_csv(path, skiprows=[0, 1, 2, 4, 5], header=[0, 1])

    # 建立坐标系
    # CS = CoordinateSystem()
    # # Ankle coordinate system
    # lAnk = dat.loc[:, "NMB:R_LANK"].values
    # mAnk = dat.loc[:, "NMB:R_MANK"].values
    # iAnk = 0.5 * (lAnk + mAnk)
    # fbk = dat.loc[:, "NMB:R_FBK"].values
    #
    # foot_CS = CoordinateSystem(iAnk, mAnk, fbk)
    # foot_CS.main()
    # e_foot = foot_CS.e
    # foot_angular_velocity = foot_CS.angular_velocity
    # plt.plot(foot_angular_velocity)
    # plt.show()

    # TibiaFibula coordinate system

    lTep = dat.loc[:, "NMB:R_LTEP"].values
    mTep = dat.loc[:, "NMB:R_MTEP"].values
    iTep = 0.5 * (lTep + mTep)
    tib = dat.loc[:, "NMB:R_TIB"].values
    iTep = filterSig(iTep, 10, 360)
    mTep = filterSig(mTep, 10, 360)
    tib = filterSig(tib, 10, 360)

    shank_CS = CoordinateSystem(iTep, mTep, tib)
    shank_CS.main()
    e_shank = shank_CS.e

    shank_angular_velocity = shank_CS.angular_velocity
    plt.plot(shank_angular_velocity[1617:1728])
    plt.show()


    # shank_CS.preprocessing.filterSig()

    # A_OB = np.zeros((len(e_B), 3, 3))
    # e_O = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)
    # e_B = shank_CS.e
    # for i in range(len(e_B)):
    #     A_OB[i] = e_O @ e_B[i].T

    def angular_velocity(e):
        e_O = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
        e_B = e
        A_OB = np.zeros((len(e), 3, 3))
        A_OB_dot = np.zeros((len(e), 3, 3))
        for i in range(len(e)):
            A_OB[i] = e_O @ e_B[i].T
        for i in range(3):
            for j in range(3):
                A_OB_dot[:, i, j] = derivative5p(array=A_OB[:, i, j], delta_t=1 / 360)
        tilde_omega = np.zeros((len(e), 3, 3))
        for i in range(len(e)):
            tilde_omega[i] = A_OB[i].T @ A_OB_dot[i]
        angular_velocity = np.vstack((tilde_omega[:, 2, 1], tilde_omega[:, 0, 2], tilde_omega[:, 1, 0])).T


    def LCS(p1, p2, p3):
        v1 = p2 - p1  # first axis x
        v2 = np.cross(v1, p3 - p1)  # second axis y
        v3 = np.cross(v1, v2)  # third axis z
        v1 = np.cross(v2, v3)  # x
        v1norm = np.linalg.norm(v1, axis=1)
        v2norm = np.linalg.norm(v2, axis=1)
        v3norm = np.linalg.norm(v3, axis=1)
        # Vector normalization
        e1 = np.zeros((v1.shape))
        e2 = np.zeros((v2.shape))
        e3 = np.zeros((v3.shape))
        for i in range(3):
            e1[:, i] = v1[:, i] * (1 / v1norm)
            e2[:, i] = v2[:, i] * (1 / v2norm)
            e3[:, i] = v3[:, i] * (1 / v3norm)
        e = np.hstack((e1, e2, e3))
        return e


    def angular_velocity2(e):
        e_B = e
        e_O = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).reshape(3, 3)
        A_OB = np.zeros((e_B.shape))
        A_OB_dot = np.zeros((A_OB.shape))
        omega = np.zeros((A_OB_dot.shape))

        for i in range(len(e_B)):
            temp = e_O @ e_B[i].reshape(3, 3).T
            A_OB[i] = temp.flatten()

        for i in range(9):
            A_OB_dot[:, i] = derivative5p(A_OB[:, i], delta_t=1 / 360)
        for i in range(len(omega)):
            temp = A_OB[i].reshape(3, 3).T @ A_OB_dot[i].reshape(3, 3)
            omega[i] = temp.flatten()
        return np.vstack((omega[:, 6], omega[:, 2], omega[:, 3])).T


    iTep = filterSig(iTep, 50, 360)
    mTep = filterSig(mTep, 50, 360)
    tib = filterSig(tib, 50, 360)

    p1 = iTep
    p2 = mTep
    p3 = tib

    e = LCS(p1, p2, p3)
    omega2 = angular_velocity2(e)

    plt.plot(omega2[1617:1728])
    plt.show()
