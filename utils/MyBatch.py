from utils import dataPreprocessing
import numpy as np
from scipy import signal

TARGET_POINT_NUM = 2048
class MyBatch():
    def __init__(self,data):
        self.data = data
        self.new_logit,self.new_label = data.get_batch() # 当前获取到的batch

    def resample(self,sig, target_point_num=None):
        '''
        对原始信号进行重采样
        :param sig: 原始信号
        :param target_point_num:目标型号点数
        :return: 重采样的信号
        '''
        sig = signal.resample(sig, target_point_num) if target_point_num else sig
        return sig

    def verflip(self,sig):
        '''
        信号竖直翻转
        :param sig:
        :return:
        '''
        return sig[::-1, :]

    def shift(self,sig, interval=20):
        '''
        上下平移
        :param sig:
        :return:
        '''
        for col in range(sig.shape[1]):
            offset = np.random.choice(range(-interval, interval))
            sig[:, col] += offset
        return sig

    def scaling(self,X, sigma=0.1):
        scalingFactor = np.random.normal(loc=1.0, scale=sigma, size=(1, X.shape[1]))
        myNoise = np.matmul(np.ones((X.shape[0], 1)), scalingFactor)
        return X * myNoise

    def transform(self,sig, train=False):
        # 前置不可或缺的步骤
        sig = self.resample(sig, TARGET_POINT_NUM)
        # # 数据增强
        if train:
            if np.random.randn() > 0.5: sig = self.scaling(sig)
            if np.random.randn() > 0.5: sig = self.verflip(sig)
            if np.random.randn() > 0.5: sig = self.shift(sig)
        # 后置不可或缺的步骤
        sig = sig.transpose()
        sig = torch.tensor(sig.copy(), dtype=torch.float)
        return sig