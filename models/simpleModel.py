import numpy as np
import tensorflow as tf
from utils import config
from tensorflow.contrib import slim
from models.MyResnet import *
class SimpleModel:
     def __init__(self,is_training = True):
         self.net_name = "Resnet34"
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
             # self.accuracy = self.get_accuracy(self.logits,self.labels) # 准确度
         else:
             self.test_logits = self.map2OneHot(self.logits)

     """
     构建网络
     """
     def build_network(self,input,output_num,is_training):
         net = input
         net = tf.reshape(net,[tf.shape(net)[0],self.length,self.lead_count])
         # net = inference(net,trainable=is_training)
         # print(net.get_shape())
         # net = res_layer2d(net)
         # print(net.get_shape())
         # net = get_half(net,net.get_shape()[2])
         # print(net.get_shape())
         # net = res_block(net,5,64,5,name="b_1")
         # net = get_half(net, net.get_shape()[2])
         # net = res_block(net, 5, 128, 5,name="b_2")
         # net = get_half(net, net.get_shape()[2])
         # net = res_block(net, 5, 256, 5,name="b_3")
         # net = get_half(net, net.get_shape()[2])
         # net = res_block(net, 10, 380, 5,name="b_4")
         # print(net.get_shape())



         #以下是2d版本
         # net = tf.reshape(net,[tf.shape(net)[0],self.lead_count,self.length,1]) # 将输入变成符合conv2d输入的shape
         # net = inference(net, is_training)
         # net = inference(net) # ResNet34 的卷积层（去掉最后一层池化）
         net = slim.flatten(net)
         net = slim.fully_connected(net, 1024, trainable=is_training)
         # net = slim.fully_connected(net,256, trainable=is_training)
         net = slim.fully_connected(net,output_num, trainable=is_training, activation_fn=None)
         return net
     """
        将输出的一维向量变成onehot
     """
     def map2OneHot(self,logits):
        list = []
        # list.append(logits)
        logits = tf.nn.sigmoid(logits) # 先通一层sigmoid
        one = tf.ones_like(logits)
        zero = tf.zeros_like(logits)
        temp = tf.where(logits < self.threshold, x=zero, y=one) # 将元素二值化映射，大于threshold的设置为1，反之，0
        logits = tf.subtract(logits,logits) # 将图清零
        logits = tf.add(logits,temp) # 将映射结果加入（为了获取梯度，不知道有没有更好的方法# ）
        list.append(logits)
        return logits
     """
     返回损失函数
     """
     def get_loss(self,logits,labels):

         # logits = self.map2OneHot(logits) # 映射

         # out = -tf.reduce_mean(labels*tf.log(tf.clip_by_value(logits,1e-10,1.0)))
         # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
         # return out
         return  tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits,labels=labels))
     # def get_accuracy(self,logits,labels):
     #     rst = self.map2OneHot(logits=logits) # 映射
     #     l = logits.get_shape() # batch_size
     #     print(l)
     #     accuracy = tf.Variable(initial_value=0,dtype=tf.float32)
     #     accuracy = tf.subtract(accuracy,accuracy) # 清零
     #     for i in range(l):
     #         b = tf.reduce_mean(tf.cast(tf.equal(rst[i], labels[i]),'float'))
     #         accuracy = tf.add(accuracy,b)
     #
     #     return tf.div(accuracy,l)

if __name__ == '__main__':
    model = SimpleModel(True)
