import numpy as np
import tensorflow as tf
from utils import config

slim = tf.contrib.slim
class SimpleModel:
     def __init__(self,is_training = True):
         self.lead_count = config.LEAD_COUNT # 导联数
         self.length = config.LENGTH #  数据十秒内记录的次数
         self.label_num = config.LABEL_NUM # 预测异常数量
         self.batch_size = config.BATCH_SIZE
         self.input = tf.placeholder(
             tf.float32, [None,self.lead_count,self.length],name='ECG'
         )
         self.logits = self.build_network(
             self.input,self.label_num,is_training
         )                                    # 函数未构建
         if is_training:
             self.labels = tf.placeholder(
                 tf.float32,[None,self.label_num]
             )
             self.loss = self.get_loss(self.logits,self.labels) # 函数未构建
     """
     构建网络
     """
     def build_network(self,input,output_num,is_training):
         pass
     """
     返回损失函数
     """
     def get_loss(self,logits,labels):
         pass
if __name__ == '__main__':
    model = SimpleModel(True)