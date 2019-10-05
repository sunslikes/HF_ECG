import tensorflow as tf
from utils import config
from utils.dataPreprocessing import DATA
from models.simpleModel import SimpleModel
from tensorflow.contrib import slim

if __name__ == '__main__':
    data = DATA(False)
    net = SimpleModel(True)
    global_step = tf.train.create_global_step()

    optimizer = tf.train.GradientDescentOptimizer(
        learning_rate= config.LEARNING_RATE
    )
    op = slim.learning.create_train_op(
        net.loss,optimizer,global_step
    )
    sess = tf.Session() # 会话
    sess.run(tf.global_variables_initializer()) # 初始化

    x_train,y_train = data.get_batch()
    feed_dict = {
        net.input: x_train,
        net.labels: y_train
    }
    sess.run(op,feed_dict=feed_dict)