#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/22 0:47
#@Author: 林先森
#@File  : config.py

DATA_PATH = 'D:\\良目\\天池大赛\\dataSet\\' # 数据集总目录
ARRYTHMIA_CLASSES_PATH = 'hf_round1_arrythmia.txt' # 分类文件路径
LABEL_PATH = 'hf_round1_label.txt' # 标签文件路径
TRAIN_PATH = 'train\\' # 训练集目录
CACHE_PATH = 'cache\\' # 缓存文件夹
BATCH_SIZE = 10
CACHE_SIZE = 100*BATCH_SIZE #单个缓存文件的大小
READ_STEP  = 5*CACHE_SIZE #单步读取量

LENGTH = 5000  # 数据十秒内记录的次数
LEAD_COUNT = 8 # 导联数，默认按




