import tensorflow as tf
import tensorflow.contrib.slim as slim
#双层残差模块
def res_layer2d(input_tensor,kshape = 5,deph = 64,conv_stride = 1,padding='SAME', trainable = True,keep_prob = 0.5):
    data = input_tensor


    #模块内部第一层卷积
    #data = slim.conv2d(data,num_outputs=deph,kernel_size=kshape,stride=conv_stride,padding=padding, trainable=trainable)
    data = slim.conv1d(data,num_outputs=deph,kernel_size=kshape,stride=conv_stride,padding = padding,trainable=trainable,activation_fn=None) # 无激活函数
    data = slim.batch_norm(data, activation_fn=tf.nn.relu, trainable=trainable)

    data = slim.dropout(data,keep_prob=keep_prob,is_training=trainable)
    # #模块内部第二层卷积
    # data = slim.conv2d(data,num_outputs=deph,kernel_size=kshape,stride=conv_stride,padding=padding,activation_fn=None, trainable=trainable)
    data = slim.conv1d(data, num_outputs=deph, kernel_size=kshape, stride=conv_stride, padding=padding,
                       activation_fn=None, trainable=trainable)
    data = slim.batch_norm(data, activation_fn=None, trainable=trainable)
    output_deep = input_tensor.get_shape().as_list()[2]

    #当输出深度和输入深度不相同时，进行对输入深度的全零填充
    if output_deep != deph:
        input_tensor = tf.pad(input_tensor,[[0, 0], [0, 0], [abs(deph-output_deep)//2,abs(deph-output_deep)//2] ]) #这里的变纬度，用的是论文中的A方法
    data = tf.add(data,input_tensor)
    data = tf.nn.relu(data)
    return data

# net = get_half(net,net.get_shape()[2])
def get_half(input_tensor,deph, trainable = True):
    data = input_tensor
    data = slim.conv1d(data,deph//2,1,stride = 2, trainable=trainable) #用1*1的卷积代替下采样
    return data

#组合同类残差模块
def res_block(input_tensor,kshape,deph,layer = 0,half = False,name = None, trainable=True,keep_prob = 0.5):
    data = input_tensor
    with tf.variable_scope(name):
        if half:
            data = get_half(data,deph//2, trainable=trainable)
        for i in range(layer//2):
            data = res_layer2d(input_tensor = data,deph = deph,kshape = kshape, trainable=trainable,keep_prob=keep_prob)
        return data

CONV_SIZE = 7 # 原本是 3
CONV_DEEP = 64 #8 loss下降不了
# NUM_LABELS = 10


#定义模型传递流程
def inference(input_tensor, regularizer = None, trainable=True,keep_prob = 0.5):
    with slim.arg_scope([slim.conv1d,slim.max_pool2d],stride = 1,padding = 'SAME'):

        with tf.variable_scope("layer1-initconv"):

            data = slim.conv1d(input_tensor, CONV_DEEP , 15, trainable=trainable)
            # data = slim.max_pool2d(data,[2,2],stride=2)
            data = tf.nn.max_pool1d(input=data, ksize=3, strides=2,padding='SAME')

            with tf.variable_scope("resnet_layer"):

                data = res_block(input_tensor = data,kshape = CONV_SIZE,deph = CONV_DEEP,layer = 6,half = False,name = "layer4-9-conv", trainable=trainable,keep_prob=keep_prob)
                data = res_block(input_tensor = data,kshape = CONV_SIZE,deph = CONV_DEEP * 2,layer = 8,half = True,name = "layer10-15-conv", trainable=trainable,keep_prob=keep_prob)
                data = res_block(input_tensor = data,kshape = CONV_SIZE + 4,deph = CONV_DEEP * 4,layer = 12,half = True,name = "layer16-27-conv", trainable=trainable,keep_prob=keep_prob)
                data = res_block(input_tensor = data,kshape = CONV_SIZE,deph = CONV_DEEP * 8,layer = 6,half = True,name = "layer28-33-conv", trainable=trainable,keep_prob=keep_prob)

                # data = slim.avg_pool2d(data,[2,2],stride=2) # 此时tensor的shape是：[10,1,313,512]，无法继续池化.
                # data = tf.layers.average_pooling1d(inputs=data, pool_size=3, strides=2, padding=1)
                data = tf.layers.average_pooling1d(inputs=data, pool_size=data.shape.as_list()[1],
                                                   strides=1)  # (batch_siaze,1,512)
                return data