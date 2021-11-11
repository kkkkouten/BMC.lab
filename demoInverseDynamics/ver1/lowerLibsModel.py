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

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/inertialMass.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/mass.py")
sys.path.append("/FM/FM3/coorinateSystem.py")
sys.path.append("/FM/FM3/preprocessing.py")

from mass import Mass
from inertialMass import InertialMass
from CoordinateSystem import Kinematics
from demo import basicProcess

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class LowerLibsModel(object):
    def __init__(self, dat, fp, weight, gender="male"):
        self.dat = dat
        self.fp = fp
        self.weight = weight
        self.gender = gender
        self.dt = 1 / 360

    def main(self):
        self.filterAndInterpolate()
        self.transFp2Motive()
        self.load_constant_param()
        self.load_param()
        self.load_velocity_accleration()
        self.load_angularVelocity_angularAccleration()
        self.load_equation_of_motion()

    def filterAndInterpolate(self):
        basis = basicProcess()
        Dat = []
        for i in range(len(self.dat.columns)):
            if i in [0, 1]:
                pass
            else:
                temp = basis.filterSig(self.dat.iloc[:, i].values, 50, 1800)
                Dat.append(temp)
        Dat = pd.DataFrame(Dat).T
        Dat.columns = self.dat.columns[2:]
        self.dat = Dat
        FP = []
        for i in range(3, 9):
            temp = basis.filterSig(self.fp.iloc[:, i].values, 50, 1800)
            temp = basis.fitFreq(temp, 1800, 360)
            FP.append(temp)
        FP = pd.DataFrame(FP).T
        FP.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        self.fp = FP

    def transFp2Motive(self):
        basis = basicProcess()
        A_fp2motive = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)
        F = self.fp.iloc[:, :3].values.T
        F = A_fp2motive @ F
        self.F = F.T

        M = self.fp.iloc[:, 3:6].values.T
        M = A_fp2motive @ M
        self.M = M.T

        p_cop = basis.getCOP(self.fp).T
        p_cop = A_fp2motive @ p_cop
        self.p_cop = p_cop.T

    def load_constant_param(self):
        g = [0, 9.8, 0] * len(self.dat)
        self.g = np.array(g).reshape(-1, 3)

    def load_param(self):
        mass = Mass()
        I = InertialMass()
        kinema = Kinematics(self.dat)
        mass.fit(self.weight, gender=self.gender)
        I.fit(self.weight)
        self.m_foot = mass.foot
        self.m_ratio_foot = mass.footRatio
        self.I_foot = I.foot.reshape(-1, 1)
        self.m_shank = mass.shank
        self.m_ratio_shank = mass.shankRatio
        self.I_shank = I.shank.reshape(-1, 1)
        # foot_ratio = np.array([17.7, 8.8, 18.2]).reshape(-1, 1)
        # shank_ratio = np.array([27.4, 27.1, 9.7]).reshape(-1, 1)
        self.cal = self.dat.loc[:, "NMB:R_CAL"].values
        self.toe = self.dat.loc[:, "NMB:R_TOE"].values
        self.lAnk = self.dat.loc[:, "NMB:R_LANK"].values
        self.mAnk = self.dat.loc[:, "NMB:R_MANK"].values
        self.fbk = self.dat.loc[:, "NMB:R_FBK"].values
        self.lTep = self.dat.loc[:, "NMB:R_LTEP"].values
        self.mTep = self.dat.loc[:, "NMB:R_MTEP"].values
        self.iTep = 0.5 * (self.lTep + self.mTep)
        self.tib = self.dat.loc[:, "NMB:R_TIB"].values
        self.iAnk = 0.5 * (self.lAnk + self.mAnk)
        self.e_foot = kinema.localFrame(self.iAnk, self.mAnk, self.fbk)
        self.e_shank = kinema.localFrame(self.iTep, self.mTep, self.tib)
        self.p_foot_proximal = self.cal
        self.p_foot_distal = self.toe
        self.p_shank_proximal = 0.5 * (self.lTep + self.mTep)
        self.p_shank_distal = 0.5 * (self.lAnk + self.mAnk)

    def load_velocity_accleration(self):
        # velocity & acceleration
        basis = basicProcess()
        # foot
        self.p_foot_com = self.p_foot_proximal + self.m_ratio_foot * (self.p_foot_distal - self.p_foot_proximal)
        self.v_foot = basis.derivative5p(self.p_foot_com, order=1, delta_t=1 / 360)
        self.a_foot = basis.derivative5p(self.p_foot_com, order=2, delta_t=1 / 360)
        # shank
        self.p_shank_com = self.p_shank_proximal + self.m_ratio_shank * (self.p_shank_distal - self.p_shank_proximal)
        self.v_shank = basis.derivative5p(self.p_shank_com, order=1, delta_t=1 / 360)
        self.a_shank = basis.derivative5p(self.p_shank_com, order=2, delta_t=1 / 360)

    def load_angularVelocity_angularAccleration(self):
        # angular velocity & angular acceleration
        # foot
        kinema = Kinematics(self.dat)
        e_foot_list = self.e_foot.tolist()
        e_foot_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.e_foot = np.array(e_foot_list)
        self.theta_1order_foot = kinema.getOmega(self.e_foot)
        theta_1order_foot_temp = np.insert(self.theta_1order_foot, 0, [0, 0, 0], axis=0)
        self.theta_2order_foot = np.diff(theta_1order_foot_temp, axis=0) / self.dt
        # shank
        e_shank_list = self.e_shank.tolist()
        e_shank_list.append([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
        self.e_shank = np.array(e_shank_list)
        self.theta_1order_shank = kinema.getOmega(self.e_shank)
        theta_1order_shank_temp = np.insert(self.theta_1order_shank, 0, [0, 0, 0], axis=0)
        self.theta_2order_shank = np.diff(theta_1order_shank_temp, axis=0) / self.dt

    def load_equation_of_motion(self):
        p_origin_fp = np.array([[0.2,0.3,0]*len(self.p_cop)]).reshape(-1,3)
        self.l_cop = p_origin_fp + self.p_cop - self.p_shank_proximal
        self.l_foot = self.p_foot_com - self.p_shank_proximal
        self.l_shank = self.p_shank_com - self.p_shank_proximal

        self.torque_foot_by_g = np.cross(self.l_foot, self.m_foot * self.g)
        self.torque_foot_by_acc = np.cross(self.l_foot, self.a_foot)
        self.torque_foot = (self.I_foot * self.theta_2order_foot.reshape(3, -1)).reshape(-1, 3) + \
                           np.cross(self.theta_1order_foot,
                                    (self.I_foot * self.theta_1order_foot.reshape(3, -1)).reshape(-1, 3))

        self.torque_shank_by_g = np.cross(self.l_shank, self.m_shank * self.g)
        self.torque_shank_by_acc = np.cross(self.l_shank, self.a_shank)
        self.torque_shank = (self.I_shank * self.theta_2order_shank.reshape(3, -1)).reshape(-1, 3) + \
                            np.cross(self.theta_1order_shank,
                                     (self.I_shank * self.theta_1order_shank.reshape(3, -1)).reshape(-1, 3))
        self.GRF_component = np.cross(self.l_cop, self.F)
        trans = np.vstack((np.zeros(len(self.dat)),np.ones(len(self.dat)),np.zeros(len(self.dat)))).T
        self.free_moment = trans*( self.M[:,1] - self.GRF_component[:,1]).reshape(-1,1)
        self.torque = self.torque_shank + self.torque_foot\
                    + self.torque_shank_by_acc + self.torque_foot_by_acc \
                    - self.torque_shank_by_g - self.torque_foot_by_g \
                    + self.GRF_component \
                    - self.free_moment


if __name__ == "__main__":
    path_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp/NMBOW20201127_001_forceplate_1.csv"

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
    # llm.main()

    torque = llm.torque
    GRF_component = llm.GRF_component
    fm = llm.free_moment

    plt.plot(torque[1617:1728, 2], color="b")
    # plt.plot(GRF_component[1617:1728, :], color="r")
    plt.plot(fm[1617:1728, 2], color="g")
    plt.show()

    plt.plot(llm.F[:,2])
    plt.show()




