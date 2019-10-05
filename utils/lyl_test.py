from models.simpleModel import SimpleModel
from utils.dataPreprocessing import DATA
import tensorflow as tf
def text_model():
    data = DATA(False)
    net = SimpleModel(True)
    x_train , y_train  = data.get_batch()
    op = net.map2OneHot(net.logits)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        result = sess.run(op,feed_dict={net.input:x_train , net.labels :y_train})
    print(result)
text_model()