#!/usr/bin/env python
# -*- coding:utf-8 -*-
#@Time  : 2019/9/27 1:04
#@Author: 林先森
#@File  : model.py

import utils.config as config

class Test():
    def __init__(self):
        self.test_path = config.DATA_PATH + config.TEST_PATH
        self.answer_file = config.DATA_PATH + config.ANSWAR_PATH

