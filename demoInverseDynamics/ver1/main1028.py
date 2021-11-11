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
from scipy import interpolate
from scipy import signal
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/inertialMass.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/mass.py")
sys.path.append("/FM/FM3/coorinateSystem.py")
sys.path.append("/FM/FM3/preprocessing.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/lowerLibsModel.py")

from mass import Mass
from inertialMass import InertialMass
from CoordinateSystem import Kinematics
from demo import basicProcess
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


def fit(x,coef,intercept):
    return x*coef + intercept

def linearModel(x,y):
    reg = LinearRegression().fit(x.reshape(-1, 1), y.reshape(-1, 1))
    # 画回归线
    fitx = x
    fitx.sort()
    fity = fit(fitx, reg.coef_, reg.intercept_)
    fity = fity.flatten()
    return fitx,fity



def main():
    files_mocap = os.listdir(path_folder_mocap)
    files_fp = os.listdir(path_folder_fp)
    files_mocap = del_DS_Store(files_mocap)
    files_fp = del_DS_Store(files_fp)
    files_mocap.sort()
    files_fp.sort()
    for file_mocap,file_fp in zip(files_mocap,files_fp):
        path_mocap = os.path.join(path_folder_mocap,file_mocap)
        path_fp = os.path.join(path_folder_fp, file_fp)
        dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
        fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
        fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']
        weight = 70
        llm = LowerLibsModel(dat, fp, weight=weight, gender="male")
        llm.filterAndInterpolate()
        llm.transFp2Motive()
        llm.load_constant_param()
        llm.load_param()
        llm.load_velocity_accleration()
        llm.load_angularVelocity_angularAccleration()
        llm.load_equation_of_motion()

        sta,end = llm.fp.Fz[100:][llm.fp.Fz[100:]>10].index[0],llm.fp.Fz[100:][llm.fp.Fz[100:]>10].index[-1]
        # plt.subplot(211)
        # plt.plot(llm.torque[sta:end,2],c="b",label="knee moment due to GRF")
        l = end-sta
        xx = np.array([i/l for i in range(l)])
        ind = np.where(xx<0.1)[0].tolist()
        y = llm.free_moment[sta:end, 1] / weight
        x = llm.GRF_component[sta:end, 1] / weight
        plt.subplot(211)
        plt.plot(y, c="b", label="free moment")
        plt.plot(x,c="r",label="GRF component")
        plt.grid(linestyle="-.")
        plt.subplot(212)
        plt.scatter(x[ind],y[ind],c="b")
        fitx,fity = linearModel(x[ind], y[ind])
        plt.plot(fitx,fity,c="r")
        plt.xlabel("vertical knee moment due to GRF ")
        plt.ylabel("free moment")
        plt.grid(linestyle="-.")


if __name__ == "__main__":
    # 导入数据 1127
    # L_45 rear
    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp"
    main()
    plt.show()
    # L_45 fore
    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_38"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/fore"
    main()
    plt.show()



    # 导入数据 1125
    # L_45 rear
    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/L_45_rear"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/rear"
    main()
    plt.show()
    # L_45 fore
    path_folder_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/L_45"
    path_folder_fp = "/Users/kkkkouten/BMC.lab/NMB/20201125/NMBUR20201125_MotiveLabeledCsv/fore"
    main()
    plt.show()






