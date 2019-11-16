#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/23 1:34
#@Author: 林先森
#@File  : test.py

# import utils.config as config
# def test_read_file():
#     f = open(config.DATA_PATH + config.TEXT_FILE)
#     count = 0
#     data_list = []
#     for line in f:
#         #第一行不读取
#         if count == 0:
#             count += 1
#             continue
#         list = []
#         list = line.split()
#         data_list.append(list)
#     print(data_list)
#
# def test_get_map():
#     f = open(config.DATA_PATH + config.ARRYTHMIA_CLASSES_PATH, encoding="UTF-8")
#     classes = []
#     for line in f:
#         classes += line.split()
#     return classes
#
# import numpy as np
# def test_get_label():
#     classes = test_get_map()
#     file = open(config.DATA_PATH + config.LABEL_PATH,  encoding="UTF-8")
#     for line in file:
#         # 初始化标签
#         label = np.zeros(len(classes))
#         list = line.split()
#         print(list[0])
#         list.pop(0)
#         for word in list:
#             if word in classes:
#                 label[classes.index(word)] = 1
#         print(label)
#
# import numpy as np
# def testnumpy():
#     test = np.zeros((4,3))
#     test[1, 0] = 1
#     print(test)

from models.simpleModel import SimpleModel
from utils.dataPreprocessing import DATA
import tensorflow as tf
def text_model():
    data = DATA(True)
    net = SimpleModel(False)
    x_train , y_train  = data.get_batch()
    op = net.map2OneHot(net.logits)
    with tf.Session as sess:
        print(sess.run(op,feed_dict={net.input:x_train , net.labels :y_train}))
text_model()