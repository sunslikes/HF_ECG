#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/27 1:04
#@Author: 林先森
#@File  : model.py

import utils.config as config
import utils.dataPreprocessing as dp
import os
import pickle
import math
import numpy as np
import tensorflow as tf

class Test():
    def __init__(self, rebuild):
        self.test_path = config.DATA_PATH + config.TEST_PATH # 测试集目录
        self.answer_file = config.DATA_PATH + config.ANSWAR_PATH
        self.cache_path = config.DATA_PATH + config.TEST_CACHE + 'data\\'# 测试集数据缓存路径
        self.cache_size = config.CACHE_SIZE
        if rebuild:
            self.rebuilding()
        self.cursor = 0  # 指针
        self.batch_size = 1
        self.cache_block_num = 0  # 缓存块号
        self.tdata_cache = self.load_cache(0)

    """
        第一次数据预处理
    """
    def rebuilding(self):
        ECG_pathList = self.getTest()  # ECG文件路径列表
        length = len(ECG_pathList)
        remain = length
        print("len:" + str(length))
        for i in range(0, math.ceil(length / self.cache_size)):
            _from = (i) * self.cache_size
            if i < math.ceil(length / self.cache_size) -1:
                to = (i + 1) * self.cache_size
            else:
                to = _from + remain
            ECG_Batch = dp.DATA(False).get_ECG_batch(ECG_pathList[_from:to])
            data_base_name = self.cache_path + 'data\\tdata_'
            base_cache_name = 'cache_'
            cache_name = base_cache_name + str(_from) + "_" + str(to)
            with open(data_base_name + cache_name, 'wb') as f:
                pickle.dump(ECG_Batch, f)
            print("缓存第" + str(_from) + "到第" + str(to) + "条数据")
            remain -= self.cache_size # 剩余数量

    '''
        获得测试集的文件列表
    '''
    def getTest(self):
        ECG_paths = []
        fileNumber = 0
        for root, sub_folders, files in os.walk(self.test_path):  # os.walk返回一个三元组(root,dirs,files)
            for name in files:  # 存ECG文件路径
                fileNumber += 1
                ECG_paths.append(os.path.join(root, name))
        return ECG_paths  # 先返回前一千个 test

    """
        载入缓存中的数据
        _from ; 从第几条数据开始，读取cache_size个数据，必须是cache_size 的整数倍
        return : 返回 cache_size 大小的数据
    """
    def load_cache(self, _from):
        print(_from)
        data_base_name = self.cache_path
        data = []
        for root, sub_folders, files in os.walk(data_base_name):  # os.walk返回一个三元组(root,dirs,files)
            for name in files:
                list = name.split('_')[-2]
                if name.split('_')[-2] == str(_from):
                    file = open(os.path.join(root, name), 'rb')
                    data = pickle.load(file)
                    break
        return data

    def get_batch(self):
        cache_block_num = math.floor((self.cursor) / self.cache_size)  # 计算当前缓存块号码
        cache_cur = self.cursor % self.cache_size  # 计算块内指针
        self.cursor += self.batch_size  # 指针迭代
        if (cache_block_num == self.cache_block_num):
            if cache_block_num == 8:
                print(str(cache_block_num) + " " + str(cache_cur), end=' ')
            return self.tdata_cache[cache_cur:cache_cur + self.batch_size]
        else:
            self.cache_block_num = cache_block_num
            self.tdata_cache= self.load_cache(cache_block_num * self.cache_size)
            return self.tdata_cache[cache_cur:cache_cur + self.batch_size]

if __name__ == '__main__':
    test = Test(False)
    for i in range(0, 8036):
        test.get_batch()
