#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/22 0:47
#@Author: 林先森
#@File  : config.py
import os

MODEL_NAME = '' # 模型名称 用于保存

DATA_PATH = 'C:/Users/Admin/PycharmProjects/HF_ECG/dataSet/' # 数据集总目录
ARRYTHMIA_CLASSES_PATH = 'hf_round1_arrythmia.txt' # 分类文件路径
LABEL_PATH = 'hf_round1_label.txt' # 标签文件路径
TRAIN_PATH = 'train/' # 训练集目录（需要手动创建）
CACHE_PATH = 'cache/' # 缓存文件夹（需要手动创建）
TEST_PATH = 'test\\testA\\' # 测试集文件夹
TEST_CACHE = 'test\\cache\\'
ANSWAR_PATH = 'test\\hf_round1_subA.txt' # 试卷
ANSWERS_DIR = 'test\\answers\\' # 测试结果目录
WEIGHTS_DIR = os.path.join(DATA_PATH,'weights') # 模型文件夹
WEIGHTS_FILE = None # 迁移模型
# WEIGHTS_FILE = 'D:\\良目\\天池大赛\\dataSet\\output\\2019_10_04_23_36\\ECG-100'                          # 迁移模型
# WEIGHTS_FILE = os.path.join(WEIGHTS_DIR,'ECG-100.data-00000-of-00001')  # 迁移模型
OUTPUT_DIR = os.path.join(DATA_PATH, 'output') # 模型输出文件夹

BATCH_SIZE = 10
CACHE_SIZE = 100*BATCH_SIZE #单个缓存文件的大小
READ_STEP  = 5*CACHE_SIZE #单步读取量

LENGTH = 5000  # 数据十秒内记录的次数
LEAD_COUNT = 8 # 导联数，默认按
LABEL_NUM = 55 # 异常标签数

LEARNING_RATE = 0.02 # 学习率 0.05
DECAY_STEPS = 30000 # 
DECAY_RATE = 0.1 # 衰退率
STAIRCASE = True # 若为true，每DECAY_STEPS次迭代更新学习率，反之，每次迭代都更新
THRESHOLD = 0.5 # sigmoid函数出来超过这个值将映射为1

SUMMARY_ITER = 100 # 每训练SUMMARY_ITER个批次进行一次评估（计算loss，输出log）
SAVE_ITER = 1000  # 每训练SAVE_ITER个批次进行模型的保存
MAX_ITER = 20000   # 训练批次




