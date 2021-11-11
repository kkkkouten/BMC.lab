#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/09/30 21:12

@author: Tei Koten
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

sys.path.append("/FM/FM3/other/demoImage.py")
sys.path.append("/FM/FM3/preprocessing.py")


from FM.FM3.other.demoImage import Image
from preprocessing import basicProcess


plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


def del_DS_Store(alist):
    if '.DS_Store' in alist:
        del_index = alist.index('.DS_Store')
        del alist[del_index]
    if '.DS_S.txt' in alist:
        del_index = alist.index('.DS_S.txt')
        del alist[del_index]
    return alist


def sortKey(x):
    try:
        res = np.array(x[:2]).astype(np.int).tolist()
    except:
        res = np.array(x[:1]).astype(np.int).tolist()
    return res

def getCOP(Mx, My, Fz):
    COP_x = My / Fz
    COP_y = -Mx / Fz
    return np.array([COP_x, COP_y]).T

def getFreeMoment(COP, Fx, Fy, Mz):
    COP_x, COP_y = COP[:, 0], COP[:, 1]
    FM = Mz - Fy * COP_x + Fx * COP_y
    return FM

def fit(x,coef,intercept):
    return x*coef + intercept

def mainNums(folderPath):
    files = os.listdir(folderPath)
    files = del_DS_Store(files)
    files.sort()
    Nums = []
    for file in files:
        path = os.path.join(folderPath,file)
        imgs = Image(path)
        dat = imgs.readImage()
        index = imgs.findStaEnd(dat)
        nums = imgs.countNonzeroElem(dat)
        nums = nums[np.where(nums > 0)[0]]
        Nums.append(nums)
    Nums = pd.DataFrame(Nums).T
    return Nums

def mainFM(folderPath):
    files = os.listdir(folderPath)
    files = del_DS_Store(files)
    files.sort(key=sortKey)
    FM = []
    for file in files[:12]:
        path = os.path.join(folderPath,file)
        dat = pd.read_table(path, sep=",")
        bp = basicProcess()
        dat, ans = bp.batchProcess(dat, 40, 2000)  # 其中包含转换为100Hz
        ind = bp.findSync(ans)
        res = ans.iloc[ind[0]:ind[1], :]
        res.index = [i for i in range(len(res))]
        cop = getCOP(res.Mx,res.My,res.Fz)
        fm = getFreeMoment(cop,res.Fx,res.Fy,res.Mz)
        FM.append(fm)
    FM = pd.DataFrame(FM).T
    return FM

def returenIndex(AB):
    A = []
    for ab in AB:
        for i in range(11):
            x = [j / len(Nums.iloc[:,i]) * 100 for j in range(len(Nums.iloc[:,i]))]
            x = np.array(x)
            start  = np.where(x>=ab[0])[0][0]
            end = np.where(x<=ab[1])[0][-1]
            if start not in A:
                A.append(start)
            if end not in A:
                A.append(end)
    return A

def generateAB(a):
    res = []
    for i in range(len(a)):
        if i == 0:
            left = a[0]
            continue
        else:
            right = a[i]
        res.append([left,right])
        left = right
    return res


if __name__ == "__main__":

    # iScan 100Hz
    folderPath = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_iscan_csv/L_45"
    Nums = mainNums(folderPath)

    # labchart 2000Hz
    folderPath = "/Users/kkkkouten/BMC.lab/NMB/20201127/processed"
    FM = mainFM(folderPath)


    # time-series data of frictional moment and pressure sensor data
    plt.figure(figsize=(10,5))
    plt.subplot(211)
    for i in range(11):
        x = [j / len(FM.iloc[:, i]) * 100 for j in range(len(FM.iloc[:, i]))]
        plt.scatter(x, FM.iloc[:,i],edgecolors='black',s=30,c="b",alpha=0.5)
        plt.plot(x,FM.iloc[:,i],c="black",alpha=0.5)
    plt.xticks([i * 10 for i in range(11)],
               ["{}%".format(i * 10) for i in range(11)])
    plt.title("frictional momoment")
    plt.ylabel("N*m")
    # plt.legend()
    plt.grid(linestyle="-.")
    plt.subplot(212)
    for i in range(11):
        x = [j / len(Nums.iloc[:,i]) * 100 for j in range(len(Nums.iloc[:,i]))]
        plt.scatter(x, Nums.iloc[:, i],edgecolors='black',s=30,c="b",alpha=0.5)
        plt.plot(x,Nums.iloc[:, i], c="black", alpha=0.5)
    plt.xticks([k * 10 for k in range(11)],
               ["{}%".format(k * 10) for k in range(11)])
    plt.title("count of pressure sensor seuare")
    plt.ylabel("count")
    # plt.legend()
    plt.grid(linestyle="-.")
    plt.tight_layout()
    plt.show()
    #
    FM.dropna(axis=0, how="any", inplace=True)
    FM.index = [i for i in range(len(FM))]
    Nums.dropna(axis=0, how="any", inplace=True)
    Nums.index = [i for i in range(len(Nums))]

    if len(FM) < len(Nums):
        Nums = Nums.iloc[:len(FM),:11]
    if len(FM) > len(Nums):
        FM = FM.iloc[:len(Nums),:11]

    FM=  FM.iloc[:,:11]

    # AB = [[0, 10], [10, 20], [20, 30], [30, 40], [40, 50],
    #       [50, 60], [60, 70], [70, 80], [80, 90], [90, 100]]

    a = np.arange(0,105,25)
    # a = [0,30,]
    AB = generateAB(a)
    AB = [[0,40],[40,50],[50,75],[75,100]]
    A = returenIndex(AB)  # 索引值
    A = generateAB(A)
    A = [[0,11],[11,13],[13,19],[19,25]]



    #  correlation coeffience
    plt.figure(figsize=(10, 10))
    for j in range(len(A)):
        try:
            plt.subplot(221 + j)
            x = Nums.iloc[A[j][0]:A[j][1], :11].values.reshape(-1, 1)
            y = FM.iloc[A[j][0]:A[j][1], :11].values.reshape(-1, 1)
            y = np.abs(y)
            # fitting linear model
            X = np.hstack((x, y))
            X = X[~np.isnan(X).any(axis=1), :]
            reg = LinearRegression().fit(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))
            pho = pearsonr(X[:, 0], X[:, 1])
            # 画回归线
            fitx = X[:, 0]
            fitx.sort()
            fity = fit(fitx, reg.coef_, reg.intercept_)
            fity = fity.flatten()
            # plot line
            plt.scatter(x,y,color="b")
            plt.plot(fitx,fity,linewidth=3,c="r",label="corrcoef:"+str(round(pho[0],3)))
            plt.title(str(AB[j]))
            plt.xlabel("count of pressure sensor seuare[count]")
            plt.ylabel("frictional moment[N*m]")
            plt.legend()
        except:
            break
    plt.tight_layout()
    plt.savefig("/Users/kkkkouten/Desktop/2.png")
    plt.show()



    # A = [[0,25]]
    plt.figure(figsize=(10, 6))
    for j in range(len(A)):
        plt.subplot(221 + j)
        x = Nums.iloc[A[j][0]:A[j][1], :].values.reshape(-1, 1)
        y = FM.iloc[A[j][0]:A[j][1], :].values.reshape(-1, 1)
        y = np.abs(y)
        # fitting linear model
        X = np.hstack((x, y))
        # X = X[~np.isnan(X).any(axis=1), :]
        reg = LinearRegression().fit(X[:, 0].reshape(-1, 1), X[:, 1].reshape(-1, 1))
        pho = pearsonr(X[:, 0], X[:, 1])
        # 画回归线
        fitx = X[:, 0]
        fitx.sort()
        fity = fit(fitx, reg.coef_, reg.intercept_)
        fity = fity.flatten()
        # plot line
        plt.scatter(x, y, color="b")
        plt.plot(fitx, fity, linewidth=3, c="r", label="corrcoef:" + str(round(pho[0], 3)))
        # plt.title(str(AB[j]))
        plt.xlabel("count of pressure sensor seuare[count]")
        plt.ylabel("frictional moment[N*m]")
        plt.legend()
    plt.tight_layout()
    plt.savefig("/Users/kkkkouten/Desktop/3.png")
    plt.show()

