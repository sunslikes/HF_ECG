import numpy as np
import tensorflow as tf
from utils import config
from tensorflow.contrib import slim
class SimpleModel:
     def __init__(self,is_training = True):
         self.lead_count = config.LEAD_COUNT # 导联数
         self.length = config.LENGTH #  数据十秒内记录的次数
         self.label_num = config.LABEL_NUM # 预测异常数量
         self.threshold = config.THRESHOLD # 阈值，超过该值则映射为1
         self.batch_size = config.BATCH_SIZE
         self.input = tf.placeholder(
             tf.float32, [None,self.lead_count,self.length]
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
         net = slim.flatten(input)
         net = slim.fully_connected(net,8000)
         net = slim.fully_connected(net, 2048)
         net = slim.fully_connected(net,512)
         net = slim.fully_connected(net,output_num)
         return net
     """
        将输出的一维向量变成onehot
     """
     def map2OneHot(self,logits):
        #list = []
        logits = tf.sigmoid(logits) # 将模型输出的结果通入sigmoid函数
        #list.append(logits)
        # one = tf.ones_like(logits)
        # zero = tf.zeros_like(logits)
        # logits= tf.where(logits < self.threshold, x=zero, y=one) # 将元素二值化映射，大于threshold的设置为1，反之，0
        # list.append(logits)
        return logits
     """
     返回损失函数
     """
     def get_loss(self,logits,labels):
         logits = self.map2OneHot(logits) # 映射
         out = -tf.reduce_mean(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
         # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
         return out
