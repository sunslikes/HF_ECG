import tensorflow as tf
from utils import config as cfg
# from utils.dataPreprocessing import DATA
from utils.MyData import DATA
import os
import numpy as np

from models.simpleModel import SimpleModel
# class singleTest():
#     def __init__(self):
#         self.data = dataPreprocessing.DATA(False)
#         self.path = os.path.join(cfg.OUTPUT_DIR, 'Resnet34-2019_11_26_23_20')
#         with tf.Graph().as_default() as g:
#             self.model = SimpleModel(False)
#             self.sess = tf.Session()
#             with self.sess as sess:
#                 self.saver = tf.train.Saver()
#                 self.ckpt = tf.train.get_checkpoint_state(self.path)
#                 print("加载ckpt：" + str(self.ckpt))
#                 self.saver.restore(self.sess, self.path)
#
#
#         self.op = self.model.test_logits
#
#         # print("loading model from " + self.model_path +'......')
#     def test_a_betch(self):
#         x,_ = self.data.get_batch()
#         print(_)
#         print('------------------')
#         # rst = self.sess.run(self.op,feed_dict={self.model.input:x})
#         # print(rst)
# if __name__ == '__main__':
#     np.set_printoptions(threshold=1e6)
#     st = singleTest()
#     st.test_a_betch()
np.set_printoptions(threshold=1e6)
path = os.path.join(cfg.OUTPUT_DIR,'Resnet34-2019_12_05_20_26')
path = tf.train.latest_checkpoint(path)
print(path)
data = DATA()
net = SimpleModel(False)
op = net.test_logits
# data.get_batch()
# data.get_batch()
# data.get_batch()
# data.get_batch()
# data.get_batch('train')
# data.get_batch('train')
# data.get_batch('train')
data.train_point = 1000
data.batch_size = 1000
x,_ = data.get_batch('test')
index = data.train_index[data.train_point:data.train_point+data.batch_size]
print("====================================" + str(index))
# x,_ = data.get_batch()
x = x.transpose(0,2,1)
print(x.dtype)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.Saver()
    saver.restore(sess,path)
    rst = sess.run(op,feed_dict={net.input:x})
print(rst)
print('===============================')
print(_)
accuracy = 0
right_event = 0 # 预测正确的心电异常事件数
pre_event = 0 # 预测的事件数
event_count = 0 # 总共的事件数目
for i in range(rst.shape[0]):
    if (rst[i] == _[i]).all():
        accuracy += 1
    for j in range(rst.shape[1]):
        if abs(rst[i][j] - 1) < 0.01:
            pre_event += 1
            if  abs(rst[i][j] - _[i][j]) < 0.01:
                right_event += 1
        if abs(_[i][j] - 1) < 0.01:
            event_count += 1
try:
    _p = right_event/pre_event
except:
    _p = 0
try:
    _r = right_event/event_count
except:
    _r = 0
try:
    f1 = 2*_p*_r/(_p + _r)
except:
    f1 = 0
print("right_event:" + str(right_event))
print("pre_event:" + str(pre_event))
print("event_count:" + str(event_count))
print("p:"+str(_p))
print("r:" + str(_r))
print("f1:" + str(f1))