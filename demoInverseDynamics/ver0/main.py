#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/02 17:01

@author: Tei Koten
"""

# !/usr/bin/env python
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


def transform_x_to_y(x, A):
    y = np.zeros((len(x), 3))
    for i in range(len(x)):
        y[i] = (A[i] @ x[i].reshape(-1, 1)).reshape(1, -1)
    return y


if __name__ == "__main__":
    # file path
    path_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp/NMBOW20201127_001_forceplate_1.csv"
    # read data
    dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']

    # set weight,gender
    weight = 70
    llm = LowerLibsModel(dat, fp, weight=weight, gender="male")
    llm.main()
    # Fz > 10N
    sta = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[0]
    end = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[-1]
    #
    l = end - sta
    xx = np.array([i / l for i in range(l)]) * 100

    torque_shank = transform_x_to_y(llm.torque, llm.e_shank)/weight
    GRF_component_shank = transform_x_to_y(llm.GRF_component, llm.e_shank)/weight
    free_moment_shank = transform_x_to_y(llm.free_moment_in_motive, llm.e_shank)/weight
    angular_velocity = llm.foot_angular_veloctiy



    plt.plot(torque_shank[sta:end, 0], c="b", label=" (+) flexion (-)extension")
    plt.plot(torque_shank[sta:end, 1], c="r", label=" (+) internal rotaion (-)external rotation")
    plt.plot(torque_shank[sta:end, 2], c="g", label=" (+) varus (-)valgus")
    plt.hlines(0, 0, 100, color="black")
    plt.ylabel("N*m/kg")
    plt.title("Resultant moment transfrom to shank")
    plt.legend()
    plt.show()

    plt.plot(GRF_component_shank[sta:end, 0], c="b", label=" (+) flexion (-)extension")
    plt.plot(GRF_component_shank[sta:end, 1], c="r", label=" (+) internal rotaion (-)external rotation")
    plt.plot(GRF_component_shank[sta:end, 2], c="g", label=" (+) varus (-)valgus")
    plt.hlines(0, 0, 100, color="black")
    plt.ylabel("N*m/kg")
    plt.title("GRF_component transform to shank")
    plt.legend()
    plt.show()

    plt.plot(free_moment_shank[sta:end, 0], c="b", label=" (+) flexion (-)extension")
    plt.plot(free_moment_shank[sta:end, 1], c="r", label=" (+) internal rotaion (-)external rotation")
    plt.plot(free_moment_shank[sta:end, 2], c="g", label=" (+) varus (-)valgus")
    plt.hlines(0, 0, 100, color="black")
    plt.ylabel("N*m/kg")
    plt.title("free moment transform to shank")
    plt.legend()
    plt.show()

















