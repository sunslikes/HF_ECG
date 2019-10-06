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
import tensorflow as tf
import datetime
from models.simpleModel import SimpleModel

class Test():
    def __init__(self, rebuild):
        self.data = dp.DATA(False) # 获取疾病种类和其中其它的方法
        self.test_path = config.DATA_PATH + config.TEST_PATH # 测试集目录
        self.answer_file = config.DATA_PATH + config.ANSWAR_PATH
        self.cache_path = config.DATA_PATH + config.TEST_CACHE + 'data\\'# 测试集数据缓存路径
        self.cache_size = config.CACHE_SIZE
        self.answers_dir = config.DATA_PATH + config.ANSWERS_DIR
        if rebuild:  # 是否重新写入函数
            self.rebuilding()
        self.cursor = 0  # 指针
        self.batch_size = 1
        self.cache_block_num = 0  # 缓存块号
        self.tdata_cache = self.load_cache(0)
        self.net = SimpleModel(False)
        self.model_file = 'SimpleModel-2019_10_07_02_15'
        self.model_dir = os.path.join(config.OUTPUT_DIR, self.model_file)

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
            ECG_Batch = self.data.get_ECG_batch(ECG_pathList[_from:to])
            if not os.path.exists(self.cache_path + 'data'):
                os.makedirs(self.cache_path + 'data')
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
            return self.tdata_cache[cache_cur:cache_cur + self.batch_size]
        else:
            self.cache_block_num = cache_block_num
            self.tdata_cache= self.load_cache(cache_block_num * self.cache_size)
            return self.tdata_cache[cache_cur:cache_cur + self.batch_size]

    def test(self):
        with tf.Session() as sess:
            saver = tf.train.Saver()
            if not os.path.exists(self.model_dir):
                print("滚去跑模型")
                return
            model_file = tf.train.latest_checkpoint(self.model_dir)
            saver.restore(sess, model_file) # 加载模型
            print("重载训练模型")
            ls = os.listdir(self.test_path)
            count = 0 # 计算有多少个数据
            for i in ls:
                if os.path.isfile(os.path.join(self.test_path, i)):
                    count += 1
            results = []
            for i in range(0, count):
                x = self.get_batch()
                y = sess.run([self.net.test_logits], feed_dict={self.net.input: x})
                results.append(y[0].tolist()[0])
            # 得到预测的onthots,接下来转成文本并保存
            sample = open(self.answer_file, encoding='utf-8')
            if not os.path.exists(self.answers_dir):
                os.makedirs(self.answers_dir)
            answer = open(os.path.join(self.answers_dir , self.model_file.split('-')[0] + str(datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))+'.txt'), 'w', encoding='utf-8')
            count = 0
            for line in sample:
                newline = line.split('\n')[0]
                number = 0 # 第几类
                for hot in results[count]:
                    if abs(hot - 1) < 1e-3:
                        newline = newline + '\t' + self.data.classes[number]
                    number += 1
                newline += '\n'
                print(newline)
                answer.write(newline)
                count += 1
            print("测试结束，请在" + os.path.realpath(answer) + '查看')


if __name__ == '__main__':
    test = Test(False)
    test.test()
