import numpy as np
import tensorflow as tf
from utils import config
from tensorflow.contrib import slim
from models.Resnet34 import inference
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
         net = input
         # net = tf.reshape(net,[self.batch_size,self.lead_count,self.length,1]) # 将输入变成符合conv2d输入的shape
         # net = inference(net) # ResNet34 的卷积层（去掉最后一层池化）
         net = slim.flatten(net)
         net = slim.fully_connected(net, 1024)
         net = slim.fully_connected(net,256)
         net = slim.fully_connected(net,output_num)
         return net
     """
        将输出的一维向量变成onehot
     """
     def map2OneHot(self,logits):
        # list = []
        logits = tf.sigmoid(logits) # 将模型输出的结果通入sigmoid函数
        # list.append(logits)
        one = tf.ones_like(logits)
        zero = tf.zeros_like(logits)
        temp = tf.where(logits < self.threshold, x=zero, y=one) # 将元素二值化映射，大于threshold的设置为1，反之，0
        logits = tf.subtract(logits,logits) # 将图清零
        logits = tf.add(logits,temp) # 将映射结果加入（为了获取梯度，不知道有没有更好的方法# ）
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

if __name__ == '__main__':
    model = SimpleModel(True)