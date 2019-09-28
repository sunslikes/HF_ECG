#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/23 1:02
#@Author: 林先森
#@File  : dataPreprocessing.py

import utils.config as CONFIG
import numpy as np
import os

class DATA:
    def __init__(self):
        self.data_path = CONFIG.DATA_PATH  # 数据集总目录
        self.train_path = self.data_path + CONFIG.TRAIN_PATH # 训练集目录
        self.classes_path = self.data_path + CONFIG.ARRYTHMIA_CLASSES_PATH # 分类文件路径
        self.label_path = self.data_path + CONFIG.LABEL_PATH # 标签文件路径
        self.data_length = CONFIG.LENGTH # ECG十秒内记录的次数
        self.lead_count = CONFIG.LEAD_COUNT # 导联数
        self.classes = self.get_classes() # 心率疾病的分类列表
        self.ECG_pathList, self.ECG_labelList = self.getTrain()  # ECG文件路径列表和对应的标签列表（打乱并排序完成）
        self.ECG_Batch = self.get_ECG_batch(self.ECG_pathList) # ECG 24106*5000*8的输出矩阵

    #获取心率疾病种类
    def get_classes(self):
        file = open(self.classes_path,  encoding="UTF-8")
        classes = []
        for line in file:
            classes += line.split()
        return classes

    '''
    获取ECG文件路径列表（打乱）和对应的标签列表
    '''
    def getTrain(self):
        ECG_paths = []
        fileNumber = 0
        for root, sub_folders, files in os.walk(self.train_path):  # os.walk返回一个三元组(root,dirs,files)
            for name in files:  # 存ECG文件路径
                fileNumber += 1
                ECG_paths.append(os.path.join(root, name))
        labels = np.zeros((fileNumber, len(self.classes)))
        file = open(self.label_path, encoding="UTF-8")
        for line in file:
            # 初始化标签
            list = line.split()
            fileName = list[0] #取得ECG文件名称
            fileName = self.train_path + fileName
            labelNumber = ECG_paths.index(fileName)
            for word in list:
                if word in self.classes:
                    one = self.classes.index(word)
                    labels[labelNumber, self.classes.index(word)] = 1
        labels_list = labels.tolist()
        temp = np.array([ECG_paths, labels_list])
        temp = temp.transpose()
        np.random.shuffle(temp)
        ECG_paths = temp[:, 0].tolist()
        labels_list = temp[:, 1].tolist()
        return ECG_paths, labels_list

    '''
    #读取ECG文件，得到一个5000*8的numpy
    '''
    def read_ECGfile(self, file_path):
        f = open(file_path, encoding="UTF-8")
        count = 0
        data_list = []
        for line in f:
            # 第一行不读取
            if count == 0:
                count += 1
                continue
            list_lead = line.split() # 同个时刻导联们的计数
            list_lead = list(map(int, list_lead)) # 转成int
            data_list.append(list_lead)
        data_np = np.array(data_list)
        data_np = data_np.transpose()
        return data_np


    def get_ECG_batch(self, ECG_paths):
        ECG_temp = []
        count = 1
        for file_path in ECG_paths:
            print(str(count) + "    " + file_path)
            count += 1
            data_numpy = self.read_ECGfile(file_path)
            data_list = data_numpy.tolist()
            ECG_temp.append(data_list)
        ECG_Batch = np.asarray(ECG_temp)
        return ECG_Batch


if __name__ == '__main__':
    VOC = DATA()
    print(VOC.ECG_Batch)
    print(VOC.ECG_Batch.shape)


