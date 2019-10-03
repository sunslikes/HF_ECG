import os
import datetime
import tensorflow as tf
from models.simpleModel import SimpleModel
from utils import config
from utils.timer import Timer
from utils.dataPreprocessing import DATA
import tensorflow.contrib.slim as slim

class Solver():

    def __init__(self,net,data):
        self.net = net  # 网络类
        self.data = data #  数据类
        self.weight_file = config.WEIGHTS_FILE # 迁移模型
        self.initial_learing_rate = config.LEARNING_RATE # 学习率
        self.decay_steps = config.DECAY_STEPS # 学习率衰退，更新频率相关量
        self.decay_rate = config.DECAY_RATE   # 衰退率
        self.staircase = config.STAIRCASE    # 布尔值，详情查看config.py
        self.summary_iter = config.SUMMARY_ITER # 评估周期
        self.save_iter = config.SAVE_ITER # 保存模型周期
        self.output_dir = os.path.join(
            config.OUTPUT_DIR, datetime.datetime.now().strftime('%Y_%m_%d_%H_%M')) # 本次模型输出的跟文件夹
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)      # 若上方文件夹不存在，创建
        self.save_cfg() # 储存本次训练配置文件信息



    """
        保存本次训练的配置信息
    """
    def save_cfg(self):
        print("保存配置信息到: "+ os.path.join(self.output_dir, 'config.txt').__str__().replace('\\','/'))
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = config.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)

if __name__ == '__main__':
    data = DATA(False)
    net = SimpleModel()
    solver = Solver(net,data)
