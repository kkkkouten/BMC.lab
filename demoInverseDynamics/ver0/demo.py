#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/08 15:23

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





if __name__=="__main__":

    path_mocap = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45/NMBOW20201127_001.csv"
    path_fp = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_MotiveLabeledCsv/L_45_fp/NMBOW20201127_001_forceplate_1.csv"

    dat = pd.read_csv(path_mocap, skiprows=[0, 1, 2, 4, 5], header=[0, 1])
    fp = pd.read_csv(path_fp, skiprows=[i for i in range(32)], header=[0])
    fp.columns = ['MocapFrame', 'MocapTime', 'DeviceFrame', 'Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz', 'Cx', 'Cy', 'Cz']

    res = []
    for i in range(len(dat.columns)):
        if i in [0, 1]:
            pass
        else:
            temp = preprocessing.filterSig(dat.iloc[:, i].values, 10, 1800)
            res.append(temp)
    res = pd.DataFrame(res).T
    res.columns = dat.columns[2:]
    dat = res


    res = []
    for i in range(3, 9):
        temp = preprocessing.filterSig(fp.iloc[:, i].values, 50, 1800)
        temp = preprocessing.fitFreq(temp, 1800, 360)
        res.append(temp)
    res = pd.DataFrame(res).T
    res.columns = ['Fx', 'Fy', 'Fz', 'Mx', 'My', 'Mz']
    fp = res

    
    
    weight = 70
    gender = "male"
    
    preprocessing = preprocessing()
    mass = Mass()
    inertia_moment = InertialMass()
    mass.fit(weight, gender=gender)
    inertia_moment.fit(weight)

    gravity = np.array([0, -9.8, 0] * len(dat)).reshape(-1, 3)  # gravity in motive CS
    r_fp_origin_in_motive = np.array([[0.2, 0, 0.3] * len(dat)]).reshape(-1, 3)
    mat_fp_to_motive = np.array([1, 0, 0, 0, 0, 1, 0, -1, 0]).reshape(3, 3)
    mat_inertial_moment_to_motive = np.array([1, 0, 0, 0, 0, 1, 0, 1, 0]).reshape(3, 3)

    force_fp = fp.iloc[:, :3].values.T
    force_motive = mat_fp_to_motive @ force_fp
    force_motive = force_motive.T
    # motive
    moment_fp = fp.iloc[:, 3:6].values.T
    moment_motive = mat_fp_to_motive @ moment_fp
    moment_motvie = moment_motive.T
    
    

    m_foot = mass.foot
    m_ratio_foot = mass.footRatio
    I_foot = mat_inertial_moment_to_motive @ inertia_moment.foot.reshape(-1, 1)
    # shank
    m_shank = mass.shank
    m_ratio_shank = mass.shankRatio
    I_shank = mat_inertial_moment_to_motive @ inertia_moment.shank.reshape(-1, 1)
    # postion
    cal = dat.loc[:, "NMB:R_CAL"].values
    toe = dat.loc[:, "NMB:R_TOE"].values
    lAnk = dat.loc[:, "NMB:R_LANK"].values
    mAnk = dat.loc[:, "NMB:R_MANK"].values
    fbk = dat.loc[:, "NMB:R_FBK"].values
    lTep = dat.loc[:, "NMB:R_LTEP"].values
    mTep = dat.loc[:, "NMB:R_MTEP"].values
    tib = dat.loc[:, "NMB:R_TIB"].values
    iTep = 0.5 * (lTep + mTep)
    iAnk = 0.5 * (lAnk + mAnk)

    foot_cs = CoordinateSystem(iAnk, mAnk, fbk)
    shank_cs = CoordinateSystem(iTep, mTep, tib)
    foot_cs.main()
    shank_cs.main()
    e_foot = foot_cs.e
    e_shank = shank_cs.e
    r_foot_proximal = cal
    r_foot_distal = toe
    r_shank_proximal = 0.5 * (lTep + mTep)
    r_shank_distal = 0.5 * (lAnk + mAnk)
    

    r_foot_com = r_foot_proximal + m_ratio_foot * (r_foot_distal - r_foot_proximal)
    v_foot = preprocessing.derivative5p(array=r_foot_com, order=1, delta_t=1 / 360)
    a_foot = preprocessing.derivative5p(array=r_foot_com, order=2, delta_t=1 / 360)
    # shank
    r_shank_com = r_shank_proximal + m_ratio_shank * (r_shank_distal - r_shank_proximal)
    v_shank = preprocessing.derivative5p(array=r_shank_com, order=1, delta_t=1 / 360)
    a_shank = preprocessing.derivative5p(array=r_shank_com, order=2, delta_t=1 / 360)

    foot_angular_veloctiy = foot_cs.angular_velocity
    foot_angular_acceleration = preprocessing.derivative5p(array=foot_angular_veloctiy, order=1,
                                                                     delta_t=1 / 360)
    # shank
    shank_angular_veloctiy = shank_cs.angular_velocity
    shank_angular_acceleration = preprocessing.derivative5p(array=shank_angular_veloctiy, order=1,
                                                                      delta_t=1 / 360)

    r_foot = r_foot_com - r_shank_proximal
    r_shank = r_shank_com - r_shank_proximal
    torque_foot_by_g = np.cross(r_foot, m_foot * gravity)
    torque_foot_by_acc = np.cross(r_foot, a_foot)
    torque_foot = (I_foot * foot_angular_veloctiy.reshape(3, -1)).reshape(-1, 3) + \
                       np.cross(foot_angular_acceleration,
                                (I_foot * foot_angular_acceleration.reshape(3, -1)).reshape(-1, 3))

    torque_shank_by_g = np.cross(r_shank, m_shank * gravity)
    torque_shank_by_acc = np.cross(r_shank, a_shank)
    torque_shank = (I_shank * shank_angular_veloctiy.reshape(3, -1)).reshape(-1, 3) + \
                        np.cross(shank_angular_acceleration,
                                 (I_shank * shank_angular_acceleration.reshape(3, -1)).reshape(-1, 3))
    GRF_component = np.cross(r_cop_motive, force_motive)
    torque = torque_shank + torque_foot \
                  + torque_shank_by_acc + torque_foot_by_acc \
                  - torque_shank_by_g - torque_foot_by_g \
                  + GRF_component \
                  - free_moment_in_motive











