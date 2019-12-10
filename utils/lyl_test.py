from models.simpleModel import SimpleModel
# from utils.dataPreprocessing import DATA
from utils.MyData import DATA
import tensorflow as tf
import numpy as np


def get_loss(logits, labels):
    out = -tf.reduce_mean(labels * tf.log(tf.clip_by_value(logits, 1e-10, 1.0)))
    # return tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    return out
def test_model():
    data = DATA()
    data.batch_size = 2 # 设置小bach_size
    net = SimpleModel(True,weight=data.loss_weight)
    x_train , y_train  = data.get_batch('train')
    x_train = x_train.transpose(0,2,1)
    logits = net.logits
    op = net.map2OneHot(net.logits)
    loss = net.loss
    show_loss = net.show_lost
    show_weighted_loss = net.show_weighted_lost
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_result,loss_result,result,show_loss_result,weited_loss_result = sess.run([logits,loss,op,show_loss,show_weighted_loss],feed_dict={net.input:x_train , net.labels :y_train})
        # result = sess.run(op,feed_dict={net.input:x_train , net.labels :y_train})
    print('logits'+ str(logits_result))
    print('map_to: ' + str(result))
    print('labels : ' + str(y_train))
    print('loss: '+ str(loss_result))
    print('show_loss' + str(show_loss_result.tolist()))
    print('weighted_loss' + str(weited_loss_result.tolist()))
    print('weight' +str(data.loss_weight.tolist()))
def test_loss():
    logits = tf.zeros([10,55])
    _ = [x for x in range(0,55)]
    list = [_ for x in range(0,10)]

    for i in range(0,55):
        for j in range(0,10):
            if list[j][i] % 2 == 0:
                list[j][i] = 1
            else:
                list[j][i] = 0

    print('list :'+ str(list))
    ones = np.ones([10,55])


    labels = np.array(list)
    logits = tf.add(logits, np.subtract(ones,labels))
    # for i in range(0,10):
    #     # list[i][0] = 1
    #     # list[i][3] = 1
    #     # if i*3 in range(5,33):
    #     #     list[i][6] = 1

    labels = np.array(list)

    loss = get_loss(logits, labels)
    print('labels is :' + str(labels))
    sess = tf.Session()
    result,loss_result = sess.run([logits,loss])
    print('logits is :'+ str(result))
    print('loss is :'+ str(loss_result))
test_model()