from models.simpleModel import SimpleModel
from utils.dataPreprocessing import DATA
import tensorflow as tf
def text_model():
    data = DATA(False)
    net = SimpleModel(True)
    x_train , y_train  = data.get_batch()
    logits = net.logits
    op = net.map2OneHot(net.logits)
    loss = net.loss
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        logits_result,loss_result,result = sess.run([logits,loss,op],feed_dict={net.input:x_train , net.labels :y_train})
        # result = sess.run(op,feed_dict={net.input:x_train , net.labels :y_train})
    print(loss_result)
    print(result)
    print('loss: '+ str(loss_result))
text_model()