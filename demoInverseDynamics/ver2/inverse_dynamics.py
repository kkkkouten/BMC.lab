#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/08 18:35

@author: Tei Koten
"""

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

# sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/inertialMass.py")
# sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/CoordinateSystem.py")

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/ver2/preprocessing.py")
sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/ver2/mass.py.py")

# from inertialMass import InertialMass
from mass import Mass
from preprocessing import preprocessing

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class fp_motive_preproceesing(object):
    def __init__(self, motive, fp):
        self.motive = motive
        self.fp = fp

    def motive_preprocessing(self, cutoff, freq):
        res = preprocessing().filterSig(self.motive.iloc[:, 2:].values, filterfreq=cutoff, fzfreq=freq)
        res = pd.DataFrame(res)
        res.columns = self.motive.columns[2:]
        self.motive = res

    def fp_preprocessing(self, cutoff, freq,inverse=True):
        res = preprocessing().filterSig(self.fp.iloc[:, 3:9].values, filterfreq=cutoff, fzfreq=freq)
        temp = np.zeros((len(self.motive), 6))
        for i in range(6):
            temp[:, i] = preprocessing().fitFreq(res[:, i], 1800, 360)
        if inverse:
            temp = -1*temp # GRF数据反向
        res = pd.DataFrame(temp)
        res.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
        self.fp = res


class LoadParameters(object):

    def __init__(self, motive):
        self.motive = motive

    def load(self, project_name="NMB", dominant="R"):
        try:
            index = project_name + ":" + dominant + "_"
        except:
            raise ValueError("example project_name='NMB',dominant='R'")
        # foot
        self.toe = self.motive.loc[:, index + "TOE"].values
        self.lMp = self.motive.loc[:, index + "LMP"].values
        self.mMp = self.motive.loc[:, index + "MMP"].values
        self.fbk = self.motive.loc[:, index + "FBK"].values
        self.lAnk = self.motive.loc[:, index + "LANK"].values
        self.mAnk = self.motive.loc[:, index + "MANK"].values
        self.cal = self.motive.loc[:, index + "CAL"].values
        # shank
        self.tib = self.motive.loc[:, index + "TIB"].values
        self.lTep = self.motive.loc[:, index + "LTEP"].values
        self.mTep = self.motive.loc[:, index + "MTEP"].values
        #
        self.iMp = 0.5 * (self.lMp + self.mMp)
        self.iAnk = 0.5 * (self.lAnk + self.mAnk)
        self.iTep = 0.5 * (self.lTep + self.mTep)
        #



class CoordinateSystem(object):

    def __init__(self, p1, p2, p3):
        self.p1 = p1
        self.p2 = p2
        self.p3 = p3
        self.dt = 1 / 360
        self.preprocessing = preprocessing()

    def main(self):
        self.vectrix()
        self.local_coordinate_system()

    def vectrix(self):
        p12 = self.p2 - self.p1
        p13 = self.p3 - self.p1
        e_x = np.cross(p13, p12)
        e_z = np.cross(e_x, p12)  # z, frontal plane， 上方 +， 下方 -
        e_y = np.cross(e_z, e_x)  # y, horizontal plane, x 外侧 +， 内侧 -
        e_x = np.cross(e_y, e_z)  # x, sagittal plane y, 前方 +， 后方 -
        e_x_norm = np.linalg.norm(e_x, axis=1)
        e_y_norm = np.linalg.norm(e_y, axis=1)
        e_z_norm = np.linalg.norm(e_z, axis=1)
        for i in range(3):
            e_x[:, i] = e_x[:, i] * (1 / e_x_norm)
            e_y[:, i] = e_y[:, i] * (1 / e_y_norm)
            e_z[:, i] = e_z[:, i] * (1 / e_z_norm)
        self.e_x = e_x
        self.e_y = e_y
        self.e_z = e_z

    def local_coordinate_system(self):
        e = list()
        for i in range(len(self.p1)):
            e.append(np.vstack((self.e_x[i], self.e_y[i], self.e_z[i])).T)
        self.e = np.array(e)


class Rotation(object):
    def __init__(self, e):
        self.e = e

    def main(self):
        self.angular_velocity()
        self.angular_acceleration()

    def angular_velocity(self):
        # e_O = np.array([-1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)
        e_B = self.e
        A_OB = e_B.T
        A_OB_T = A_OB.T
        A_OB_dot = np.zeros((A_OB.shape))
        tilde_omega = np.zeros((3, 3, len(e_B)))
        # 微分
        for i in range(3):
            for j in range(3):
                A_OB_dot[i, j, :] = preprocessing().derivative5p(A_OB[i, j, :], delta_t=1 / 360)
        # poisson formula
        for i in range(len(e_B)):
            tilde_omega[:, :, i] = A_OB_T[i, :, :] @ A_OB_dot[:, :, i]
        self.angular_velocity = np.vstack((tilde_omega[2, 1, :], tilde_omega[0, 2, :], tilde_omega[1, 0, :])).T

    def angular_acceleration(self):
        self.angular_acceleration = preprocessing().derivative5p(array=self.angular_velocity, order=1,
                                                                 delta_t=1 / 360)


class Translation(object):
    """
    r_proximal = r_p
    r_distal = r_d
    """

    def __init__(self, r_p, r_d, mass_ratio):
        self.r_p = r_p
        self.r_d = r_d
        self.mass_ratio = mass_ratio

    def main(self):
        self.center_of_mass()
        self.velocity()
        self.acceleration()

    def center_of_mass(self):
        self.r_com = self.r_p + self.mass_ratio * (self.r_d - self.r_p)

    def velocity(self):
        self.velocity = preprocessing().derivative5p(self.r_com, order=1, delta_t=1 / 360)

    def acceleration(self):
        self.acceleration = preprocessing().derivative5p(self.r_com, order=2, delta_t=1 / 360)


if __name__ == "__main__":
    # set file path
    path_motive = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp/NMBOW20201127_001_forceplate_1.csv"
    # read file
    motive = pd.read_csv(path_motive, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    # filter, interpolation
    pre = fp_motive_preproceesing(motive, fp)
    pre.motive_preprocessing(cutoff=10, freq=360)
    pre.fp_preprocessing(cutoff=10, freq=360,inverse=False)
    motive = pre.motive
    fp = pre.fp
    plt.plot(fp.iloc[1617:1728,:3])
    plt.show()


    # set motive parameters
    param = LoadParameters(motive)
    param.load()
    lTep = param.lTep
    mTep = param.mTep
    iTep = param.iTep
    iAnk = param.iAnk
    iMp = param.iMp
    mAnk = param.mAnk
    cal = param.cal
    toe = param.toe

    # create local coordinate system
    shank_CS = CoordinateSystem(iTep, mTep, iAnk)
    shank_CS.main()
    e_shank = shank_CS.e

    foot_CS = CoordinateSystem(iAnk,mAnk,iMp)
    foot_CS.main()
    e_foot = foot_CS.e

    # rotation
    shank_rotation = Rotation(e_shank)
    shank_rotation.main()
    shank_angular_velocity = shank_rotation.angular_velocity
    shank_angular_acceleration = shank_rotation.angular_acceleration

    foot_rotation = Rotation(e_foot)
    foot_rotation.main()
    foot_angular_velocity = foot_rotation.angular_velocity
    foot_angular_acceleration = foot_rotation.angular_acceleration


    # translation
    weight = 70
    mass = Mass()
    mass.fit(weight, gender="male")
    m_ratio_shank = mass.shankRatio
    m_ratio_foot = mass.footRatio

    shank_translation = Translation(iTep, iAnk, m_ratio_shank)
    shank_translation.main()
    velocity = shank_translation.velocity
    acceleration = shank_translation.acceleration

    foot_translation = Translation(cal, toe, m_ratio_foot)
    foot_translation.main()
    foot_velocity = foot_translation.velocity
    foot_acceleration = foot_translation.acceleration

    r_p_foot =
    r_d_foot =
    r_p_shank =
    r_d_shank =









