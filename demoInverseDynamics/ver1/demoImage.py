#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/02/12 19:52

@author: Tei Koten
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from scipy.ndimage.filters import gaussian_filter

plt.rcParams['xtick.direction'] = 'in'
plt.rcParams['ytick.direction'] = 'in'


class Image(object):

    def __init__(self, path):
        self.path = path

    def readImage(self):
        img = self.readFile(self.path)
        img = self.formatClean(img)
        return img

    def readFile(self, path, encoding="ISO-8859-1"):
        """
        :param encoding: utf-8 可能会用上
        :return: 包括字符串和浮点数数据
        """
        with open(path, "r", encoding=encoding) as f:
            lines = f.readlines()
        content = []
        for line in lines:
            content.append(line.strip())
        return content

    def formatClean(self, content):
        frames = []
        i, j = 0, 0
        for index, line in enumerate(content):
            if len(line) == 0 or line == "@@":
                i = j
                j = index + 1

                if i == 0:
                    pass
                elif j > i + 2:
                    sub_frame = []
                    for each in content[i + 1: j - 1]:
                        sub_frame.append(each.split(","))
                    frames.append(np.array(sub_frame).astype(np.int))
        return np.array(frames)

    def findStaEnd(self, content):
        """
        寻找数据的起始点与截止点
        :param array_3d:
        :return: 数据的开始索引与结束索引
        """
        m, n, q = content.shape
        index = []
        for i in range(m):
            sum = np.sum(content[i])
            if sum != 0:
                index.append(i)
            else:
                pass
        return index

    def countNonzeroElem(self, content):
        return np.array([np.count_nonzero(i) for i in content])

    def gifChart(self, content, index):
        N = len(content)
        M = int(np.ceil(np.max(np.concatenate(content)) / 10) * 10)  # 计算颜色的极值
        fig, ax = plt.subplots(figsize=(10, 10))

        def update(curr_frm):
            if curr_frm == N:
                a.event_source.stop()
            print("当前绘制第{}帧".format(curr_frm))
            matrix = content[curr_frm]

            plt.clf()
            plt.imshow(matrix, cmap=plt.cm.hot_r, vmin=0, vmax=M)
            plt.text(0.1, 0.95, "Frame " + str(curr_frm + 1), transform=ax.transAxes, size=20, color='#777777')
            plt.colorbar()
            ax.xaxis.set_ticks_position('top')
            plt.xticks(range(44))
            plt.yticks(range(44))
            plt.title("Area of Pressure sensor")

            frame = plt.gca()
            frame.axes.get_yaxis().set_visible(True)
            frame.axes.get_xaxis().set_visible(True)

        # 100毫秒1帧，可自由调整
        # 如果想删除前后的无用帧，修改frames参数
        a = animation.FuncAnimation(fig, update, frames=range(index[0], index[-1]), interval=100)
        # a = animation.FuncAnimation(fig, update, frames=range(0, N), interval=100)
        return a




if __name__ == "__main__":
    # iScan 100Hz
    path = "/Users/kkkkouten/BMC.lab/NMB/20201127/NMBOW20201127_iscan_csv/L_45/NMBOW20201127_001_M.csv"
    imgs = Image(path)
    dat = imgs.readImage()
    index = imgs.findStaEnd(dat)
    nums = imgs.countNonzeroElem(dat)

    plt.plot(nums)
    plt.show()

    plt.imshow(dat[240])
    plt.ylabel("row")
    plt.xlabel("col")
    plt.colorbar()
    plt.show()