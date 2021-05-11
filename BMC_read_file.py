#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on 2020/12/24 19:28

@author: Tei Koten
"""

import numpy as np
import pandas as pd
from nptdms import TdmsFile


def read_file(file, encoding="IS-8859-1"):
    """
    :param file:
    :param encoding: 可能会用上 gbk, ISO-8859-1
    :return:
    """
    with open(file, "r", encoding=encoding) as f:
        lines = f.readlines()
    content = []
    for line in lines:
        content.append(line.strip())
    return content


def clean_format(DeviceData, type='DAQ'):
    """pass"""

    DAQ_operators = [224.8089439, 224.8089439, 2231.039289,  # fp2 fx,fy,fz
                     885.0748065, 885.0748065, 885.0748065,  # fp2 mx,my,mz
                     224.8089439, 224.8089439, 2231.039289,  # fp1 fx,fy,fz
                     885.0748065, 885.0748065, 885.0748065]  # fp1 mx,my,mz
    if type == 'DAQ':
        columns_name = DeviceData[14].split(',')
        DeviceArray = []
        for Device in DeviceData[15:]:
            Device = np.array(Device.split(',')).astype(np.float)
            DeviceArray.append(Device)
        loss = pd.DataFrame(np.array(DeviceArray), columns=columns_name)
        # 将信号乘上系数转化为newton单位
        res = pd.DataFrame()
        for col, operator in zip(loss.columns[3:-1], DAQ_operators):
            df = loss.loc[:, col] * operator
            res = pd.concat((res, df), axis=1)
        res = pd.concat((loss.iloc[:, :3], res), axis=1)
        res = pd.concat((res, loss.iloc[:, -1]), axis=1)

    elif type == 'AMTI':
        columns_name = DeviceData[34].split(',')
        DeviceArray = []
        for Device in DeviceData[35:]:
            a = Device.split(',')  # 插入一个函数对其中的异常值进行检查
            CheckValueInList(a, '-nan(ind)', np.nan)
            CheckValueInList(a, 'nan(ind)', np.nan)
            CheckValueInList(a, 'inf', np.nan)
            CheckValueInList(a, '-inf', np.nan)
            Device = np.array(a).astype(np.float)
            DeviceArray.append(Device)
            DeviceArray.append(Device)
        res = pd.DataFrame(np.array(DeviceArray), columns=columns_name)

    elif type is "TEC":
        columns_name = ["Time", "Fx", "Fy", "Fz", "Mx", "My", "Mz"]
        data = []
        for line in content[10:]:
            string_array = line.split(',')  # 插入一个函数对其中的异常值进行检查
            array = np.array(string_array).astype(np.float)
            data.append(array)
        res = pd.DataFrame(data, columns=columns_name)

    elif type == 'ACC':
        columns_name = ['gx', 'gy', 'gz',
                        'degx', 'degy', 'degz',
                        'g2A', 'g2B',
                        'qw', 'qx', 'qy', 'qz']
        DeviceArray = []
        for Device in DeviceData[10:]:
            Device = np.array(Device.split(','))[1:-1].astype(np.float)
            DeviceArray.append(Device)
        res = pd.DataFrame(np.array(DeviceArray), columns=columns_name)
    return res


def read_tdms(path):
    """
    将单个tdms文件转化为dataframe格式，便于最后输出txt文件
    :param path: tdms_path
    :return: dataframe
    """
    tdms_file = TdmsFile.read(path)  # 读取tdms数据
    # tdms_file.groups() 　查看所有group
    tdms_groups = tdms_file.groups()[0]  # type-->list
    # tdms_groups.channels() 查看所有channels
    tdms_channels = tdms_groups.channels()
    # 分离groups中的channel数据
    ch0_data = tdms_channels[0][:]  # dual signal
    ch1_data = tdms_channels[1][:]  # fz1 or fz2
    ch2_data = tdms_channels[2][:]  # fz1 or fz2
    # 根据数据类型更换最后的索引位置
    string_ch0 = 'channel0'
    string_ch1 = 'channel1'
    string_ch2 = 'channel2'
    mat = np.vstack((ch0_data, ch1_data, ch2_data)).T  # 不转置会造成 3行n列
    sync_data = pd.DataFrame(mat, columns=[string_ch0, string_ch1, string_ch2])
    return sync_data
