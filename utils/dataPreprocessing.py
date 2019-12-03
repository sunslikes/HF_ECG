#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/23 1:02
#@Author: 林先森
#@File  : dataPreprocessing.py

import utils.config as CONFIG
import numpy as np
import os
import pickle
import math
import tensorflow as tf
import tensorflow.contrib.slim as slim

class DATA:
    """
    rebuild: 首次建立，将会读取数据集，建立缓存（或数据集发生改动时）
    """
    def __init__(self,rebuild):
        self.data_path = CONFIG.DATA_PATH  # 数据集总目录
        self.train_path = os.path.join(self.data_path, CONFIG.TRAIN_PATH)  # 训练集目录
        self.classes_path = os.path.join(self.data_path, CONFIG.ARRYTHMIA_CLASSES_PATH) # 分类文件路径
        self.label_path = os.path.join(self.data_path, CONFIG.LABEL_PATH) # 标签文件路径
        self.cache_path = os.path.join(self.data_path, CONFIG.CACHE_PATH) # 缓存文件夹路径
        self.cache_size = CONFIG.CACHE_SIZE # 单个缓存文件的大小
        self.data_length = CONFIG.LENGTH # ECG十秒内记录的次数
        self.lead_count = CONFIG.LEAD_COUNT # 导联数
        self.batch_size = CONFIG.BATCH_SIZE
        self.classes = self.get_classes() # 心率疾病的分类列表
        # self.ECG_pathList, self.ECG_labelList = self.getTrain()  # ECG文件路径列表和对应的标签列表（打乱并排序完成）
        # self.ECG_Batch = self.get_ECG_batch(self.ECG_pathList) # ECG 24106*5000*8的输出矩阵
        if rebuild:
            self.rebuilding()
        self.cursor = 0 # 指针
        self.cache_block_num = 0 # 缓存块号
        self.data_cache,self.labcel_cache = self.load_cache(0) # 缓存块
    """
        第一次数据预处理
    """
    def rebuilding(self):
        ECG_pathList, ECG_labelList = self.getTrain()  # ECG文件路径列表和对应的标签列表（打乱并排序完成）
        print('ECG文件路径列表和对应的标签列表（打乱并排序完成）')
        length = len(ECG_labelList)
        print("len:" + str(length))
        for i in range(0,math.ceil(length/self.cache_size)):
            _from = (i)*self.cache_size
            to = (i + 1) * self.cache_size
            ECG_Batch = self.get_ECG_batch(ECG_pathList[_from:to])
            self.save_cache(ECG_Batch, ECG_labelList[_from:to], _from,to)
        # ECG_Batch = self.get_ECG_batch(ECG_pathList)  # ECG 24106*5000*8的输出矩阵
        # self.save_cache(ECG_Batch,ECG_labelList,0) # 缓存
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
            fileName = os.path.join(self.train_path, fileName)
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
        return ECG_paths, labels_list # 先返回前一千个 test

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
            # print(str(count) + "    " + file_path)
            count += 1
            data_numpy = self.read_ECGfile(file_path)
            data_list = data_numpy.tolist()
            ECG_temp.append(data_list)
        ECG_Batch = np.asarray(ECG_temp)
        # print(np.size(ECG_Batch,0))
        # self.save_cache(ECG_Batch)
        return ECG_Batch
    """
        将数据与标注缓存
    """
    def save_cache(self,ECG_Batch,ECG_labelList ,_from,to):
        data_base_name = self.cache_path + 'data\\data_'
        label_base_name = self.cache_path + 'label\\label_'
        # if(ECG_labelList ==None ):
        #     ECG_labelList = self.ECG_labelList  # 要修改
        # for i in range(0,math.ceil(np.size(ECG_Batch,0)/self.cache_size)): # 储存data缓存
        #     _from = (i)*self.cache_size+start
        #     to = (i + 1) * self.cache_size+start
        #     base_cache_name = 'cache_'
        #     cache_name = base_cache_name + str(_from) + "_" + str(to)
        #     with open(data_base_name + cache_name, 'wb') as f:
        #         pickle.dump(ECG_Batch[_from:to], f)
        #     with open(label_base_name + cache_name, 'wb') as f:
        #         pickle.dump(ECG_labelList[_from:to], f)
        #     print ("缓存第"+ str(_from) + "到第" + str(to) + "条数据")
        base_cache_name = 'cache_'
        cache_name = base_cache_name + str(_from) + "_" + str(to)
        with open(data_base_name + cache_name, 'wb') as f:
            pickle.dump(ECG_Batch, f)
        with open(label_base_name + cache_name, 'wb') as f:
            pickle.dump(ECG_labelList, f)
        print("缓存第" + str(_from) + "到第" + str(to) + "条数据")

    """
    载入缓存中的数据
    _from ; 从第几条数据开始，读取cache_size个数据，必须是cache_size 的整数倍
    return : 返回 cache_size 大小的数据
    """
    def load_cache(self,_from):
        data_base_name = os.path.join(self.cache_path, 'data\\data_')
        label_base_name = os.path.join(self.cache_path, 'label\\label_')
        base_cache_name = 'cache_'
        data_path = data_base_name + base_cache_name + str(_from) + "_" + str(_from + self.cache_size)
        label_path = label_base_name + base_cache_name + str(_from) + "_" + str(_from + self.cache_size)
        fin = open(data_path,'rb')
        data = pickle.load(fin)
        # print(data)
        fin = open(label_path, 'rb')
        label = pickle.load(fin)
        # print(label)
        return data,label
    """
     返回一个batch
    """
    def get_batch(self):
        cache_block_num = math.floor((self.cursor)/self.cache_size) #计算当前缓存块号码
        cache_cur = self.cursor % self.cache_size #计算块内指针
        #self.cursor += self.batch_size # 指针迭代
        self.cursor = (self.cursor+self.batch_size)%CONFIG.DATASET_SIZE  #指针迭代

        if(cache_block_num == self.cache_block_num):
            return self.data_cache[cache_cur:cache_cur+self.batch_size],self.labcel_cache[cache_cur:cache_cur+self.batch_size]
        else:
            print('正在读取： 第'+str(cache_block_num)+'块缓存，下一个批次的指针是：' + str(self.cursor) +'块内指针是 : ' + str(cache_cur))
            #print(cache_block_num)
            self.cache_block_num = cache_block_num
            self.data_cache,self.labcel_cache = self.load_cache(cache_block_num*self.cache_size)
            return self.data_cache[cache_cur:cache_cur + self.batch_size], self.labcel_cache[
                                                                           cache_cur:cache_cur + self.batch_size]
if __name__ == '__main__':
    VOC = DATA(False)
    print(VOC.classes)
    _,y = VOC.get_batch()
    print(type(y[0][0]))
    # print(VOC.ECG_Batch)
    # print(VOC.ECG_Batch.shape)
    # with open(VOC.cache_path + 'batch','wb') as f:
    #     pickle.dump(VOC.ECG_Batch,f)
    # print(VOC.ECG_labelList[0:CONFIG.CACHE_SIZE])
    # for i in range(0,20100):
    #     x_train,y_train = VOC.get_batch()
    #     print(x_train)
    # x_train, y_train = VOC.get_batch()
    # print(np.asarray(x_train).shape)
    # print(y_train[0])






