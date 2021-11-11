#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/10/22 15:20

@author: Tei Koten
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/inertialMass.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/mass.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/CoordinateSystem.py")
sys.path.append("/FM/FM3/preprocessing.py")

from mass import Mass
from inertialMass import InertialMass
from CoordinateSystem import CoordinateSystem
from preprocessing import preprocessing

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
        self.load_constant_param()
        self.motive_preprocessing()
        self.fp_preprocessing()
        self.cop()
        self.free_moment()
        self.transform_fp_to_motive()
        self.load_param()
        self.load_local_coordinate_system()
        self.load_velocity_accleration()
        self.load_angularVelocity_angularAccleration()
        self.load_equation_of_motion()

    def load_constant_param(self):
        self.preprocessing = preprocessing()
        self.mass = Mass()
        self.inertia_moment = InertialMass()
        self.mass.fit(self.weight, gender=self.gender)
        self.inertia_moment.fit(self.weight)

        self.gravity = np.array([0, -9.8, 0] * len(self.dat)).reshape(-1, 3)  # gravity in motive CS
        self.r_fp_origin_in_motive = np.array([[0.2, 0, 0.3] * len(self.dat)]).reshape(-1, 3)
        self.mat_fp_to_motive = np.array([1, 0, 0, 0, 0, -1, 0, 1, 0]).reshape(3, 3)
        self.mat_inertial_moment_to_motive = np.array([-1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)

    def motive_preprocessing(self):
        res = []
        for i in range(len(self.dat.columns)):
            if i in [0, 1]:
                pass
            else:
                temp = self.preprocessing.filterSig(self.dat.iloc[:, i].values, 30, 1800)
                res.append(temp)
        res = pd.DataFrame(res).T
        res.columns = self.dat.columns[2:]
        self.dat = res

    def fp_preprocessing(self):
        res = []
        for i in range(3, 9):
            temp = self.preprocessing.filterSig(self.fp.iloc[:, i].values, 50, 1800)
            temp = self.preprocessing.fitFreq(temp, 1800, 360)
            res.append(temp)
        res = pd.DataFrame(res).T
        res.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        self.fp = res

    def cop(self):
        cop_x = -1 * self.fp.Mz / self.fp.Fz
        cop_y = self.fp.My / self.fp.Fz
        cop_z = [0] * len(self.fp)
        self.r_cop_fp = np.array([cop_x, cop_y, cop_z])
        self.r_cop_motive = self.r_fp_origin_in_motive + (self.mat_fp_to_motive @ self.r_cop_fp).T


    def free_moment(self):
        self.free_moment_in_fp = self.fp.iloc[:, 4].values \
                                 - self.fp.iloc[:, 1].values * self.r_cop_fp[0, :] \
                                 + self.fp.iloc[:, 0].values * self.r_cop_fp[1, :]
        self.free_moment_in_fp = np.vstack((np.zeros(len(self.dat)), np.zeros(len(self.dat)), self.free_moment_in_fp))
        self.free_moment_in_motive = self.free_moment_in_fp.T @ self.mat_fp_to_motive

    def transform_fp_to_motive(self):
        # force
        self.force_fp = -1*self.fp.iloc[:, :3].values.T
        force_motive = self.mat_fp_to_motive @ self.force_fp
        self.force_motive = force_motive.T
        # motive
        moment_fp = self.fp.iloc[:, 3:6].values.T
        moment_motive = self.mat_fp_to_motive @ moment_fp
        self.moment_motvie = moment_motive.T

    def transform_x_to_y(self, x, A):
        y = np.zeros((len(x), 3))
        for i in range(len(x)):
            y[i] = (A[i] @ x[i].reshape(-1, 1)).reshape(1, -1)
        return y

    def load_param(self):
        # foot
        self.m_foot = self.mass.foot
        self.m_ratio_foot = self.mass.footRatio
        self.I_foot = self.mat_inertial_moment_to_motive @ self.inertia_moment.foot.reshape(-1, 1)
        # shank
        self.m_shank = self.mass.shank
        self.m_ratio_shank = self.mass.shankRatio
        self.I_shank = self.mat_inertial_moment_to_motive @ self.inertia_moment.shank.reshape(-1, 1)
        # postion
        self.cal = self.dat.loc[:, "NMB:R_CAL"].values
        self.toe = self.dat.loc[:, "NMB:R_TOE"].values
        self.lAnk = self.dat.loc[:, "NMB:R_LANK"].values
        self.mAnk = self.dat.loc[:, "NMB:R_MANK"].values
        self.fbk = self.dat.loc[:, "NMB:R_FBK"].values
        self.lTep = self.dat.loc[:, "NMB:R_LTEP"].values
        self.mTep = self.dat.loc[:, "NMB:R_MTEP"].values
        self.tib = self.dat.loc[:, "NMB:R_TIB"].values
        self.iTep = 0.5 * (self.lTep + self.mTep)
        self.iAnk = 0.5 * (self.lAnk + self.mAnk)

    def load_local_coordinate_system(self):
        # foot local coordinate system & shank local coordinate system
        self.foot_cs = CoordinateSystem(self.iAnk, self.mAnk, self.fbk)
        self.shank_cs = CoordinateSystem(self.iTep, self.mTep, self.tib)
        self.foot_cs.main()
        self.shank_cs.main()
        self.e_foot = self.foot_cs.e
        self.e_shank = self.shank_cs.e
        self.r_foot_proximal = self.cal
        self.r_foot_distal = self.toe
        self.r_shank_proximal = 0.5 * (self.lTep + self.mTep)
        self.r_shank_distal = 0.5 * (self.lAnk + self.mAnk)

    def load_velocity_accleration(self):
        # velocity & acceleration
        # basis = basicProcess()
        # foot
        self.r_foot_com = self.r_foot_proximal + self.m_ratio_foot * (self.r_foot_distal - self.r_foot_proximal)
        self.v_foot = self.preprocessing.derivative5p(array=self.r_foot_com, order=1, delta_t=1 / 360)
        self.a_foot = self.preprocessing.derivative5p(array=self.r_foot_com, order=2, delta_t=1 / 360)
        # shank
        self.r_shank_com = self.r_shank_proximal + self.m_ratio_shank * (self.r_shank_distal - self.r_shank_proximal)
        self.v_shank = self.preprocessing.derivative5p(array=self.r_shank_com, order=1, delta_t=1 / 360)
        self.a_shank = self.preprocessing.derivative5p(array=self.r_shank_com, order=2, delta_t=1 / 360)

    def load_angularVelocity_angularAccleration(self):
        # angular velocity & angular acceleration
        # foot
        self.foot_angular_veloctiy = self.foot_cs.angular_velocity
        self.foot_angular_acceleration = self.preprocessing.derivative5p(array=self.foot_angular_veloctiy, order=1,
                                                                         delta_t=1 / 360)
        # shank
        self.shank_angular_veloctiy = self.shank_cs.angular_velocity
        self.shank_angular_acceleration = self.preprocessing.derivative5p(array=self.shank_angular_veloctiy, order=1,
                                                                          delta_t=1 / 360)

    def load_equation_of_motion(self):
        self.r_shank_proximal_to_cop = self.r_cop_motive - self.r_shank_proximal
        self.r_foot = self.r_foot_com - self.r_shank_proximal
        self.r_shank = self.r_shank_com - self.r_shank_proximal
        self.torque_foot_by_g = np.cross(self.r_foot, self.m_foot * self.gravity)
        self.torque_foot_by_acc = np.cross(self.r_foot, self.a_foot)
        self.torque_foot = (self.I_foot * self.foot_angular_veloctiy.reshape(3, -1)).reshape(-1, 3) + \
                           np.cross(self.foot_angular_acceleration,
                                    (self.I_foot * self.foot_angular_acceleration.reshape(3, -1)).reshape(-1, 3))

        self.torque_shank_by_g = np.cross(self.r_shank, self.m_shank * self.gravity)
        self.torque_shank_by_acc = np.cross(self.r_shank, self.a_shank)
        self.torque_shank = (self.I_shank * self.shank_angular_veloctiy.reshape(3, -1)).reshape(-1, 3) + \
                            np.cross(self.shank_angular_acceleration,
                                     (self.I_shank * self.shank_angular_acceleration.reshape(3, -1)).reshape(-1, 3))
        self.GRF_component = np.cross(self.r_shank_proximal_to_cop, self.force_motive)
        self.torque = self.torque_shank + self.torque_foot \
                      + self.torque_shank_by_acc + self.torque_foot_by_acc \
                      + self.torque_shank_by_g + self.torque_foot_by_g \
                      - self.GRF_component \
                      - self.free_moment_in_motive


if __name__ == "__main__":
    path_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp/NMBOW20201127_001_forceplate_1.csv"

    dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']

    weight = 70
    llm = LowerLibsModel(dat, fp, weight=weight, gender="male")
    llm.main()

    sta, end = llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[0], llm.fp.Fz[100:][llm.fp.Fz[100:] > 10].index[-1]
    l = end - sta
    xx = np.array([i / l for i in range(l)]) * 100

    torque = llm.transform_x_to_y(llm.torque,llm.e_shank)/weight
    GRF_component = llm.transform_x_to_y(llm.GRF_component,llm.e_shank)/weight
    free_moment = llm.transform_x_to_y(llm.free_moment_in_motive,llm.e_shank)/weight
    angular_velocity = llm.foot_angular_veloctiy

    plt.plot(xx,torque[sta:end],label="torque")
    plt.plot(xx,GRF_component[sta:end, :],label="GRF_component")
    # plt.plot(xx,free_moment[sta:end, 1], c="r",label="free moment")
    # plt.plot(angular_velocity[sta:end])
    # plt.hlines(0,0,100,color="black")
    # plt.xlabel("stance%")
    # plt.ylabel("N*m/weight + internal")
    # plt.legend()

    f_shank = llm.transform_x_to_y(llm.force_motive,llm.e_shank)
    plt.plot(f_shank[1617:1728])
    plt.show()

    # plt.plot(llm.r_cop_motive[1617:1728])
    # plt.subplot(211)
    # plt.plot(llm.force_motive[1617:1728])
    # plt.title("force motive")
    # plt.subplot(212)
    # plt.plot(llm.fp.Fx[1617:1728])
    # plt.plot(llm.fp.Fy[1617:1728])
    # plt.plot(llm.fp.Fz[1617:1728],label="Fz")
    # plt.legend()
    # plt.title("force fp")
    # plt.show()


    # cop check
    llm.r_cop_fp.T
    plt.subplot(211)
    plt.plot(llm.r_cop_motive[1617:1728,2],llm.r_cop_motive[1617:1728,0])
    plt.scatter(llm.r_cop_motive[1617, 2], llm.r_cop_motive[1617, 0])
    plt.subplot(212)
    plt.plot(llm.r_cop_fp[1,1617:1728],llm.r_cop_fp[0,1617:1728],)
    plt.show()


    llm.r_shank_proximal_to_cop

    # force
    labels = ["x","y","z"]
    plt.subplot(211)
    for i in range(3):
        plt.plot(llm.force_fp.T[1617:1728,i],label=labels[i])
    plt.legend()

    plt.subplot(212)
    for i in range(3):
        plt.plot(llm.force_motive[1617:1728,i],label=labels[i])
    plt.legend()
    plt.show()

