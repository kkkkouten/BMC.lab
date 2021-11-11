#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/10/22 15:20

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

sys.path.append("/FM/FM3/inertialMass.py")
sys.path.append("/FM/FM3/mass.py")
sys.path.append("/FM/FM3/coorinateSystem.py")
sys.path.append("/FM/FM3/preprocessing.py")

from mass import Mass
from inertialMass import InertialMass
from coorinateSystem import Kinematics
from preprocessing import basicProcess

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def getCOP(fp):
    COP_x = fp.My / fp.Fz
    COP_y = -1 * fp.Mx / fp.Fz
    COP_z = [0] * len(fp)
    return np.array([COP_x, COP_y, COP_z]).T


def getOmega(e,dt=1/360):
    e1 = e[:, 0, :]
    e2 = e[:, 1, :]
    e3 = e[:, 2, :]
    de1dt = np.diff(e1, axis=0) / dt
    de2dt = np.diff(e2, axis=0) / dt
    de3dt = np.diff(e3, axis=0) / dt
    omega1 = np.sum(de2dt * e3[0:-1, :], axis=1).reshape(-1, 1)
    omega2 = np.sum(de3dt * e1[0:-1, :], axis=1).reshape(-1, 1)
    omega3 = np.sum(de1dt * e2[0:-1, :], axis=1).reshape(-1, 1)
    omega = np.hstack((omega1, omega2, omega3))
    return omega


def filterAndInterpolate(dat, fp):
    basis = basicProcess()
    Dat = []
    for i in range(len(dat.columns)):
        if i in [0, 1]:
            pass
        else:
            temp = basis.filterSig(dat.iloc[:, i].values, 50, 1800)
            Dat.append(temp)
    Dat = pd.DataFrame(Dat).T
    Dat.columns = dat.columns[2:]
    FP = []
    for i in range(3, 9):
        temp = basis.filterSig(fp.iloc[:, i].values, 50, 1800)
        temp = basis.fitFreq(temp, 1800, 360)
        FP.append(temp)
    FP = pd.DataFrame(FP).T
    FP.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    return Dat, FP


def transFp2Motive(FP):
    A_fp2motive = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)
    F = FP.iloc[:, :3].values.T
    F = A_fp2motive @ F
    F = F.T

    M = FP.iloc[:, 3:6].values.T
    N = A_fp2motive @ M
    M = M.T

    p_cop = getCOP(FP).T
    p_cop = A_fp2motive @ p_cop
    p_cop = p_cop.T
    return F, M, p_cop


def getFootParam(dat, m_foot, m_ration_foot):
    kinema = Kinematics(dat)
    # foot coordinate system
    lAnk = dat.loc[:, "NMB:R_LANK"].values
    mAnk = dat.loc[:, "NMB:R_MANK"].values
    iAnk = 0.5 * (lAnk + mAnk)
    m1 = iAnk
    m2 = mAnk
    m3 = dat.loc[:, "NMB:R_FBK"].values
    e_foot = kinema.localFrame(m1, m2, m3)

    p_foot_proximal = dat.loc[:, "NMB:R_CAL"].values
    p_foot_distal = dat.loc[:, "NMB:R_TOE"].values
    p_foot_com = p_foot_proximal + m_ration_foot * (p_foot_distal - p_foot_proximal)
    # foot_ratio = np.array([17.7, 8.8, 18.2]).reshape(-1, 1)
    # p_foot_I = m_foot * (foot_ratio * np.linalg.norm(p_foot_distal - p_foot_proximal, axis=1)) ** 2
    # p_foot_I = p_foot_I.T
    v_foot = basis.derivative5p(p_foot_com, order=1, delta_t=1 / 360)
    a_foot = basis.derivative5p(p_foot_com, order=2, delta_t=1 / 360)
    dt = 1 / 360
    # angular velocity
    e_foot_list = e_foot.tolist()
    e_foot_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    e_foot = np.array(e_foot_list)
    theta_1order_foot = getOmega(e_foot)
    theta_1order_foot_temp = np.insert(theta_1order_foot, 0, [0, 0, 0], axis=0)
    theta_2order_foot = np.diff(theta_1order_foot_temp, axis=0) / dt
    return p_foot_proximal, p_foot_distal, p_foot_com, a_foot, theta_1order_foot, theta_2order_foot


def getShankParam(dat, m_shank, m_ration_shank):
    kinema = Kinematics(dat)
    # shank coordinate system
    lAnk = dat.loc[:, "NMB:R_LANK"].values
    mAnk = dat.loc[:, "NMB:R_MANK"].values
    lTep = dat.loc[:, "NMB:R_LTEP"].values
    mTep = dat.loc[:, "NMB:R_MTEP"].values
    iTep = 0.5 * (lTep + mTep)
    m1 = iTep
    m2 = mTep
    m3 = dat.loc[:, "NMB:R_TIB"].values
    e_shank = kinema.localFrame(m1, m2, m3)

    # shank position
    p_shank_proximal = 0.5 * (lTep + mTep)
    p_shank_distal = 0.5 * (lAnk + mAnk)
    p_shank_com = p_shank_proximal + m_ration_shank * (p_shank_distal - p_shank_proximal)
    # shank_ratio = np.array([27.4, 27.1, 9.7]).reshape(-1, 1)
    # p_shank_I = m_shank * (shank_ratio * np.linalg.norm(p_shank_distal - p_shank_proximal, axis=1)) ** 2
    # p_shank_I = p_shank_I.T
    # velocity
    v_shank = basis.derivative5p(p_shank_com, order=1, delta_t=1 / 360)
    a_shank = basis.derivative5p(p_shank_com, order=2, delta_t=1 / 360)
    # angular velocity
    dt = 1 / 360
    # angular velocity
    e_shank_list = e_shank.tolist()
    e_shank_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    e_shank = np.array(e_shank_list)
    theta_1order_shank = getOmega(e_shank)
    theta_1order_shank_temp = np.insert(theta_1order_shank, 0, [0, 0, 0], axis=0)
    theta_2order_shank = np.diff(theta_1order_shank_temp, axis=0) / dt
    return p_shank_proximal, p_shank_distal, p_shank_com, a_shank, theta_1order_shank, theta_2order_shank





if __name__ == "__main__":
    # 导入数据
    path_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001_forceplate_1.csv"

    dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']

    # motive 和 fp的滤波处理
    Dat, FP = filterAndInterpolate(dat, fp)
    F, M, p_cop = transFp2Motive(FP)
    basis = basicProcess()
    dat = Dat

    # 重力加速度
    g = [0, 9.8, 0] * len(dat)
    g = np.array(g).reshape(-1, 3)

    # 质量
    weight = 70
    mass = Mass()
    mass.fit(weight, gender="male")
    I = InertialMass()
    I.fit(weight)
    kinema = Kinematics(dat)

    # foot coordinate system
    m_foot = mass.foot
    m_ration_foot = mass.footRatio
    I_foot = I.foot.reshape(-1, 1)
    # foot_ratio = np.array([17.7, 8.8, 18.2]).reshape(-1, 1)
    dt = 1 / 360
    # foot coordinate system
    cal = dat.loc[:, "NMB:R_CAL"].values
    toe = dat.loc[:, "NMB:R_TOE"].values
    lAnk = dat.loc[:, "NMB:R_LANK"].values
    mAnk = dat.loc[:, "NMB:R_MANK"].values
    iAnk = 0.5 * (lAnk + mAnk)
    fbk = dat.loc[:, "NMB:R_FBK"].values
    e_foot = kinema.localFrame(iAnk, mAnk, fbk)
    #
    p_foot_proximal = cal
    p_foot_distal = toe
    p_foot_com = p_foot_proximal + m_ration_foot * (p_foot_distal - p_foot_proximal)
    # p_foot_I = m_foot * (foot_ratio * np.linalg.norm(p_foot_distal - p_foot_proximal, axis=1)) ** 2
    # p_foot_I = p_foot_I.T
    v_foot = basis.derivative5p(p_foot_com, order=1, delta_t=1 / 360)
    a_foot = basis.derivative5p(p_foot_com, order=2, delta_t=1 / 360)
    # angular velocity
    e_foot_list = e_foot.tolist()
    e_foot_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    e_foot = np.array(e_foot_list)
    theta_1order_foot = getOmega(e_foot)
    theta_1order_foot_temp = np.insert(theta_1order_foot, 0, [0, 0, 0], axis=0)
    theta_2order_foot = np.diff(theta_1order_foot_temp, axis=0) / dt

    # shank coordinate system
    m_shank = mass.shank
    m_ration_shank = mass.shankRatio
    I_shank = I.shank.reshape(-1, 1)
    # shank_ratio = np.array([27.4, 27.1, 9.7]).reshape(-1, 1)
    dt = 1 / 360
    #
    lTep = dat.loc[:, "NMB:R_LTEP"].values
    mTep = dat.loc[:, "NMB:R_MTEP"].values
    iTep = 0.5 * (lTep + mTep)
    tib = dat.loc[:, "NMB:R_TIB"].values
    e_shank = kinema.localFrame(iTep, mTep, tib)
    # shank position
    p_shank_proximal = 0.5 * (lTep + mTep)
    p_shank_distal = 0.5 * (lAnk + mAnk)
    p_shank_com = p_shank_proximal + m_ration_shank * (p_shank_distal - p_shank_proximal)
    # p_shank_I = m_shank * (shank_ratio * np.linalg.norm(p_shank_distal - p_shank_proximal, axis=1)) ** 2
    # p_shank_I = p_shank_I.T
    # velocity
    v_shank = basis.derivative5p(p_shank_com, order=1, delta_t=1 / 360)
    a_shank = basis.derivative5p(p_shank_com, order=2, delta_t=1 / 360)
    # angular velocity
    e_shank_list = e_shank.tolist()
    e_shank_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    e_shank = np.array(e_shank_list)
    theta_1order_shank = getOmega(e_shank)
    theta_1order_shank_temp = np.insert(theta_1order_shank, 0, [0, 0, 0], axis=0)
    theta_2order_shank = np.diff(theta_1order_shank_temp, axis=0) / dt

    # equation of motion
    l_cop = p_cop - p_shank_proximal
    l_f = p_foot_com - p_shank_proximal
    l_s = p_shank_com - p_shank_proximal

    torque_shank = (I_shank * theta_2order_shank.reshape(3, -1)).reshape(-1, 3) + \
                   np.cross(theta_1order_shank, (I_shank * theta_1order_shank.reshape(3, -1)).reshape(-1, 3))

    torque_foot = (I_foot * theta_2order_foot.reshape(3, -1)).reshape(-1, 3) + \
                  np.cross(theta_1order_foot, (I_foot * theta_1order_foot.reshape(3, -1)).reshape(-1, 3))

    GRF_component = np.cross(l_cop, F)

    torque = np.cross(l_s, m_shank * g) + np.cross(l_f, m_foot * g) \
             - np.cross(l_s, a_shank) + np.cross(l_f, a_foot) \
             + GRF_component \
             + M \
             - (torque_shank + torque_foot)

    plt.plot(torque[1617:1728, :], color="b")
    plt.plot(GRF_component[1617:1728, :], color="r")
    plt.show()

    # p_shank_proximal, p_shank_distal, p_shank_com, \
    # a_shank, theta_1order_shank, theta_2order_shank = getShankParam(dat, m_shank, m_ration_shank)
    # p_foot_proximal, p_foot_distal, p_foot_com, \
    # a_foot, theta_1order_foot, theta_2order_foot = getFootParam(dat, m_foot, m_ration_foot)
