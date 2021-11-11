#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/10/22 12:56

@author: Tei Koten
"""
import numpy as np
import sys

sys.path.append("/Users/kkkkouten/PycharmProjects/00_BMC.lab/FM/FM3/mass.py")
from mass import Mass


class InertialMass(object):

    def __init__(self):
        self.allSegment = dict()

    def fit(self, weight):
        keys = ["head", "trunk", "upperArm", "foreArm", "hand", "thigh", "shank", "foot"]
        coef = [2.14059241e-04, 1.66696758e-04, 3.17323698e-04, 2.96891978e-02, 2.84505520e-02, 1.02514936e-02,
                2.10308416e-04, 1.33594482e-04, 5.63111198e-05, 6.77942954e-05, 7.06694707e-05, 2.30144673e-05,
                8.27115577e+00, 9.89910470e+00, 2.94636165e+00, 2.03089601e-03, 2.36290977e-03, 5.39871127e-04,
                6.42501789e-04, 5.35146828e-04, 9.57895745e-05, 3.69130387e-05, 3.90331419e-05, 1.28968008e-05]
        bias = [3.13237907e-03, 5.51914997e-03, -5.97709792e-04, -3.15577317e-01, -7.66600642e-01, -2.89624007e-01,
                1.51016614e-03, 5.05986899e-03, -1.36998338e-03, 2.06495312e-03, 1.69423503e-03, -6.49929547e-04,
                1.45941208e-04, -8.84909838e-05, -1.36908766e-05, -1.72102626e-02, -3.18653729e-02, -1.39352009e-02,
                -2.37398329e-03, 4.07269742e-03, -3.37625920e-03, 9.07587559e-04, 4.99413364e-04, -8.55274976e-05]
        k = [47.9, 45.4, 36.3, 34.6, 35.7, 16.7, 26.2, 25.7, 10.7, 27.9, 27.7, 11.5,
             51.9, 57.1, 31.4, 27.8, 27.0, 15.2, 27.4, 27.1, 9.7, 17.7, 8.8, 18.2]
        k = (np.array(k) / 100).reshape(8, 3)
        coef = np.array(coef).reshape(8, 3)
        bias = np.array(bias).reshape(8, 3)
        inertialMass = coef * weight + bias
        for i in range(len(keys)):
            self.allSegment[keys[i]] = inertialMass[i, :]
        self.head = self.allSegment["head"]
        self.trunk = self.allSegment["trunk"]
        self.upperArm = self.allSegment["upperArm"]
        self.foreArm = self.allSegment["foreArm"]
        self.hand = self.allSegment["hand"]
        self.thigh = self.allSegment["thigh"]
        self.shank = self.allSegment["shank"]
        self.foot = self.allSegment["foot"]


if __name__ == "__main__":
    # example
    weight = 70
    I = InertialMass()
    I.fit(weight)
    I_foot = I.foot
    I_shank = I.shank
    I_thigh = I.thigh
