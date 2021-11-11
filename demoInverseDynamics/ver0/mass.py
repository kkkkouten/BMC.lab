#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/10/22 13:59

@author: Tei Koten
"""
import numpy as np


class Mass(object):

    def __init__(self):
        self.allSegment = dict()
        self.massRatio = dict()

    def fit(self, weight, gender="male"):
        keys = ["head", "trunk", "upperArm", "foreArm", "hand", "thigh", "shank", "foot"]
        male = np.array([6.9, 48.9, 2.7, 1.6, 0.6, 11.0, 5.1, 1.1]) / 100 * weight
        female = np.array([7.5, 45.7, 2.6, 1.5, 0.6, 12.3, 5.3, 1.1]) / 100 * weight
        maleMassRatio = np.array([82.1, 49.3, 52.9, 41.5, 89.1, 47.5, 40.6, 59.5]) / 100
        femaleMassRatio = np.array([75.9, 50.6, 52.3, 42.3, 90.8, 45.8, 41.0, 59.4]) / 100
        if gender == "male":
            for i in range(len(keys)):
                self.allSegment[keys[i]] = male[i]
                self.massRatio[keys[i]] = maleMassRatio[i]
        elif gender == "female":
            for i in range(len(keys)):
                self.allSegment[keys[i]] = female[i]
                self.massRatio[keys[i]] = femaleMassRatio[i]
        else:
            raise ValueError("gender only has 2 option 'male'/'female'")
        self.head, self.headRatio = self.allSegment["head"], self.massRatio["head"]
        self.trunk, self.trunkRatio = self.allSegment["trunk"], self.massRatio["trunk"]
        self.upperArm, self.upperArmRatio = self.allSegment["upperArm"], self.massRatio["upperArm"]
        self.foreArm, self.foreArmRatio = self.allSegment["foreArm"], self.massRatio["foreArm"]
        self.hand, self.handRatio = self.allSegment["hand"], self.massRatio["hand"]
        self.thigh, self.thighRatio = self.allSegment["thigh"], self.massRatio["thigh"]
        self.shank, self.shankRatio = self.allSegment["shank"], self.massRatio["shank"]
        self.foot, self.footRatio = self.allSegment["foot"], self.massRatio["foot"]


if __name__ == "__main__":
    # example
    weight = 70
    mass = Mass()
    mass.fit(weight, gender="male")
    massAllSegment = mass.allSegment
    m_foot = mass.foot
    m_ration_foot = mass.footRatio

    m_shank = mass.shank
    m_ration_shank = mass.shankRatio

    m_thigh = mass.thigh
    m_ration_thigh = mass.thighRatio
    print(m_foot, m_ration_foot)
    print(m_shank, m_ration_shank)
    print(m_thigh, m_ration_thigh)
    # print(mass.massRation(gender="male"))
