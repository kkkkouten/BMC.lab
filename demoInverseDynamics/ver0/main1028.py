#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/10/28 1:09

@author: Tei Koten
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/lowerLibsModel.py")
from lowerLibsModel import LowerLibsModel

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def del_DS_Store(alist):
    if '.DS_Store' in alist:
        del_index = alist.index('.DS_Store')
        del alist[del_index]
    if '.DS_S.txt' in alist:
        del_index = alist.index('.DS_S.txt')
        del alist[del_index]
    return alist


def fit(x, coef, intercept):
    return x * coef + intercept


def linearModel(x, y):
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    # 画回归线
    fitx = x
    fitx.sort()
    fity = fit(fitx, reg.coef_, reg.intercept_)
    fity = fity.flatten()
    return fitx, fity


def files_process(path_folder_mocap, path_folder_fp):
    files_mocap = os.listdir(path_folder_mocap)
    files_fp = os.listdir(path_folder_fp)
    files_mocap = del_DS_Store(files_mocap)
    files_fp = del_DS_Store(files_fp)
    files_mocap.sort()
    files_fp.sort()
    return files_mocap, files_fp


def read_files(path_folder_mocap, path_folder_fp, file_mocap, file_fp):
    path_mocap = os.path.join(path_folder_mocap, file_mocap)
    path_fp = os.path.join(path_folder_fp, file_fp)
    dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy',
                  'Cz']
    return dat, fp


def transform_x_to_y(x, A):
    y = np.zeros((len(x), 3))
    for i in range(len(x)):
        y[i] = (A[i] @ x[i].reshape(-1, 1)).reshape(1, -1)
    return y


if __name__ == "__main__":
    # 导入数据 1127
    # L_45 rear
    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp"
    # 1125 L_45 rear

    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/L_45_rear"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/rear"

    files_mocap, files_fp = files_process(path_folder_mocap, path_folder_fp)

    plt.figure(figsize=(10,10))
    for file_mocap, file_fp in zip(files_mocap, files_fp):
        dat, fp = read_files(path_folder_mocap, path_folder_fp, file_mocap, file_fp)
        weight = 70

        llm = LowerLibsModel(dat, fp, weight=weight, gender="male")
        llm.main()
        sta = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[0]
        end = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[-1]
        l = end - sta
        xx = np.array([i / l for i in range(l)])
        y1 = transform_x_to_y(llm.torque, llm.e_shank)
        y2 = transform_x_to_y(llm.GRF_component, llm.e_shank)
        y3 = transform_x_to_y(llm.free_moment_in_motive, llm.e_shank)
        y4 = transform_x_to_y(llm.shank_angular_veloctiy,llm.e_shank)

        plt.subplot(411)
        plt.plot(xx, (y1 / weight)[sta:end, 2])
        # plt.plot(xx,(y3/weight)[sta:end,1], c="b")
        plt.ylabel("N*m/kg")
        plt.title("Resultant moment rearfoot")
        plt.subplot(412)
        plt.plot(xx, (y2 / weight)[sta:end, 2])
        # plt.plot(xx,(y3/weight)[sta:end,1], c="b")
        plt.ylabel("N*m/kg")
        plt.title("GRF_component")
        plt.subplot(413)
        plt.plot(xx, (y3 / weight)[sta:end, 2])
        plt.title("free moment")
        plt.ylabel("N*m/kg")
        plt.subplot(414)
        plt.plot(xx, y4[sta:end, 2])
        plt.title("joint angluar velocity")
        plt.ylabel("rad/s")

    plt.tight_layout()
    plt.show()

    # files_mocap = os.listdir(path_folder_mocap)
    # files_fp = os.listdir(path_folder_fp)
    # files_mocap = del_DS_Store(files_mocap)
    # files_fp = del_DS_Store(files_fp)
    # files_mocap.sort()
    # files_fp.sort()
    # for file_mocap, file_fp in zip(files_mocap, files_fp):
    #     path_mocap = os.path.join(path_folder_mocap, file_mocap)
    #     path_fp = os.path.join(path_folder_fp, file_fp)
    #     dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    #     fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    #     fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy',
    #                   'Cz']
    #     weight = 70
    #     llm = LowerLibsModel(dat, fp, weight=weight, gender="male")
    #     llm.main()
    #
    #     sta, end = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[0], llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[-1]
    #     # plt.subplot(211)
    #     # plt.plot(llm.torque[sta:end,2],c="b",label="knee moment due to GRF")
    #     l = end - sta
    #     xx = np.array([i / l for i in range(l)])
    #     ind = np.where(xx < 0.1)[0].tolist()
    #     y = llm.free_moment[sta:end, 1] / weight
    #     x = llm.GRF_component[sta:end, 1] / weight
    #     plt.subplot(211)
    #     plt.plot(y, c="b", label="free moment")
    #     plt.plot(x, c="r", label="GRF component")
    #     plt.grid(linestyle="-.")
    #     plt.subplot(212)
    #     plt.scatter(x[ind], y[ind], c="b")
    #     fitx, fity = linearModel(x[ind], y[ind])
    #     plt.plot(fitx, fity, c="r")
    #     plt.xlabel("vertical knee moment due to GRF ")
    #     plt.ylabel("free moment")
    #     plt.grid(linestyle="-.")
