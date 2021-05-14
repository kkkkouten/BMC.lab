#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2021/04/21 16:09

@author: Tei Koten
"""

import os
import numpy as np
import pandas as pd


class Readfile(object):

    def __init__(self, file_path, save_path, col_name):
        self.file_path = file_path
        self.save_path = save_path
        self.col_name = col_name

    def split_dat(self):
        self.split_content(
            self.clean_format(),
            self.read_file())
        return None

    def read_file(self, encoding="ISO-8859-1"):
        """
        :param encoding: utf-8 可能会用上
        :return: 包括字符串和浮点数数据
        """
        with open(self.file_path, "r", encoding=encoding) as f:
            lines = f.readlines()
        content = []
        for line in lines:
            content.append(line.strip())
        return content

    def clean_format(self):
        content = self.read_file()
        num = list()
        i = 0
        while i < len(content):
            split_line = content[i].split()
            if split_line[0] == 'Interval=':
                # print(i, i + 6)
                num.append(i)
                i = i + 6  # +6 是因为0-5都是字符串
            else:
                i = i + 1
        return num

    def split_content(self, num, content):
        for i in range(len(num)):
            try:
                start, end = num[i] + 6, num[i + 1]
                res = list()
                for line in content[start:end]:
                    split_line = line.split()
                    Array = np.array(split_line).astype(np.float)
                    res.append(Array)
            except:
                start = num[i] + 6
                res = list()
                for line in content[start:]:
                    split_line = line.split()
                    Array = np.array(split_line).astype(np.float)
                    res.append(Array)
            # 如果columns变多可以修改
            output_res = pd.DataFrame(res, columns=self.col_name)
            output_path = os.path.join(self.save_path, str(i + 1) + '.txt')
            output_res.to_csv(output_path, index=False)
            print(i + 1, 'is finished')
        return None


if __name__ == "__main__":

    file_path = '/Users/kkkkouten/BMC.lab/TKT/20210512/labchart/TKT20210512_LabChart.txt'
    save_path = '/Users/kkkkouten/BMC.lab/TKT_processed/20210512/labchart'
    col_name = ['time', 'switch', 'GM', 'LQL', 'RQL', 'AnkGyroY','motive']

    rf = Readfile(file_path, save_path, col_name)
    rf.split_dat()
