#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/11/08 15:17

@author: Tei Koten
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def omega(eo, eb, typ: str, sf: int):
    # Angular velocity of B - CS from O-CS expressed in O - CS( O座標系に対するB座標系の角速度のO座標表記 )
    # ex.omega( WLD.e, SHK.e, "o", sf ): Angular velocity of SHK CS in WCS( 絶対座標系におけるSHK座標系の角速度 )
    outputType = type(eo)
    A11 = np.sum(eo[:, [0, 3, 6]] * eb[:, [0, 3, 6]], axis=1)
    A12 = np.sum(eo[:, [0, 3, 6]] * eb[:, [1, 4, 7]], axis=1)
    A13 = np.sum(eo[:, [0, 3, 6]] * eb[:, [2, 5, 8]], axis=1)
    A21 = np.sum(eo[:, [1, 4, 7]] * eb[:, [0, 3, 6]], axis=1)
    A22 = np.sum(eo[:, [1, 4, 7]] * eb[:, [1, 4, 7]], axis=1)
    A23 = np.sum(eo[:, [1, 4, 7]] * eb[:, [2, 5, 8]], axis=1)
    A31 = np.sum(eo[:, [2, 5, 8]] * eb[:, [0, 3, 6]], axis=1)
    A32 = np.sum(eo[:, [2, 5, 8]] * eb[:, [1, 4, 7]], axis=1)
    A33 = np.sum(eo[:, [2, 5, 8]] * eb[:, [2, 5, 8]], axis=1)

    Aob = np.vstack([A11, A21, A31, A12, A22, A32, A13, A23, A33]).T  # B->Oの座標変換行列
    dAob = dt5(Aob, sf)
    tAob = transpose(Aob)

    if typ == "o":
        tilde = mope(dAob, tAob, "o")  # Poisson's equ. マルチボディダイナミクス（1）
    elif typ == "b":
        tilde = mope(tAob, dAob, "o")  # Poisson's equ. マルチボディダイナミクス（1）
    else:
        return print('typ should be o/b')

    if outputType == pd.DataFrame:
        return pd.DataFrame(np.vstack(
            [(tilde[:, 7] - tilde[:, 5]) / 2, (tilde[:, 2] - tilde[:, 6]) / 2, (tilde[:, 3] - tilde[:, 1]) / 2]).T,
                            columns=["X", "Y", "Z"])
    else:
        return np.vstack(
            [(tilde[:, 7] - tilde[:, 5]) / 2, (tilde[:, 2] - tilde[:, 6]) / 2, (tilde[:, 3] - tilde[:, 1]) / 2]).T


def transpose(Matrix):
    return Matrix[:, [0, 3, 6, 1, 4, 7, 2, 5, 8]]


def mope(Mt, mt, typ):
    if typ == "t":  # 転置の場合
        a = transpose(np.array(Mt))
        b = np.array(mt)
    else:
        a = np.array(Mt)
        b = np.array(mt)

    if mt.shape[1] == 3:
        dat = np.vstack(
            [np.sum(a[:, 0:3] * b, axis=1), np.sum(a[:, 3:7] * b, axis=1), np.sum(a[:, 7:10] * b, axis=1)]).T
        if type(Mt) == pd.DataFrame:
            dat = pd.DataFrame(dat)
            dat.columns = ["X", "Y", "Z"]

    elif mt.shape[1] == 9:
        dat = np.vstack([np.sum(a[:, 0:3] * b[:, [0, 3, 6]], axis=1), np.sum(a[:, 0:3] * b[:, [1, 4, 7]], axis=1),
                         np.sum(a[:, 0:3] * b[:, [2, 5, 8]], axis=1),
                         np.sum(a[:, 3:6] * b[:, [0, 3, 6]], axis=1), np.sum(a[:, 3:6] * b[:, [1, 4, 7]], axis=1),
                         np.sum(a[:, 3:6] * b[:, [2, 5, 8]], axis=1),
                         np.sum(a[:, 6:9] * b[:, [0, 3, 6]], axis=1), np.sum(a[:, 6:9] * b[:, [1, 4, 7]], axis=1),
                         np.sum(a[:, 6:9] * b[:, [2, 5, 8]], axis=1)]).T
        if type(Mt) == pd.DataFrame:
            dat = pd.DataFrame(dat)
            dat.columns = ["exx", "exy", "exz", "eyx", "eyy", "eyz", "ezx", "ezy", "ezz"]

    return dat


def dt5(dat, fs):
    fss = 1 / fs
    fss = 12 * fss

    if type(dat) == pd.DataFrame:
        df1 = dat[dat.columns[dat.dtypes != "object"]]
        ar = np.array(df1)
    elif type(dat) == pd.Series:
        ar = np.array(dat)
    elif type(dat) == np.ndarray:
        ar = dat.copy()
    row = ar.shape[0]
    column = ar.shape[1]

    ar2 = np.zeros((row, column))
    ar2[0, :] = (-25 * ar[0, :] + 48 * ar[1, :] - 36 * ar[2, :] + 16 * ar[3, :] - 3 * ar[4, :]) / fss
    ar2[1, :] = (-3 * ar[0, :] - 10 * ar[1, :] + 18 * ar[2, :] - 6 * ar[3, :] + ar[4, :]) / fss
    for x in range(2, row - 2):
        ar2[x, :] = (ar[x - 2, :] - 8 * ar[x - 1, :] + 8 * ar[x + 1, :] - ar[x + 2, :]) / fss
    ar2[-2, :] = (-ar[-5, :] + 6 * ar[-4, :] - 18 * ar[-3, :] + 10 * ar[-2, :] + 3 * ar[-1, :]) / fss
    ar2[-1, :] = (3 * ar[-5, :] - 16 * ar[-4, :] + 36 * ar[-3, :] - 48 * ar[-2, :] + 25 * ar[-1, :]) / fss

    if type(dat) == pd.DataFrame:
        nan = dat[dat.columns[dat.dtypes == "object"]]
        ans = pd.concat([nan, pd.DataFrame(ar2)], axis=1)
        ans.columns = dat.columns
    elif type(dat) == pd.Series:
        ans = ar2
        ans.name = dat.name
    else:
        ans = ar2
    return ans

if __name__=="__main__":

    e =

    eo = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1] * 2172).reshape(-1, 9)
    eb = e

    w = omega(eo, eb, typ="o", sf=360)
    plt.plot(w[1617:1728])
    plt.show()