#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/05/06 18:33

@author: Tei Koten
"""

import numpy as np
from scipy import signal
from scipy import interpolate


class BMC_kinetics(object):
    """
    This class includes various common use tools about signal process for BMC.lab.
    This program is based on Biomechanics lab.
    parameters:
    * data:
    *
    *
    *
    *
    ### Example ###
    """

    def __init__(self, data):
        self.dat = data

    
