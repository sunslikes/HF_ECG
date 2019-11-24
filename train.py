from utils import config
import os
import datetime
import tensorflow as tf
from models.simpleModel import SimpleModel
from utils.timer import Timer
from utils.dataPreprocessing import DATA
import tensorflow.contrib.slim as slim
from utils.loger import Loger

class Solver():

    def __init__(self,net,data):
        self.net = net  # 网络类
        self.data = data #  数据类
        self.weight_file = config.WEIGHTS_FILE # 迁移模型
        self.initial_learning_rate = config.LEARNING_RATE # 初始学习率（后期会衰退）
        self.decay_steps = config.DECAY_STEPS # 学习率衰退，更新频率相关量
        self.decay_rate = config.DECAY_RATE   # 衰退率
        self.staircase = config.STAIRCASE    # 布尔值，详情查看config.py
        self.summary_iter = config.SUMMARY_ITER # 评估周期
        self.save_iter = config.SAVE_ITER # 保存模型周期
        self.max_iter = config.MAX_ITER   # 训练的批次数
        self.output_dir = os.path.join(
            config.OUTPUT_DIR, self.net.net_name + '-' + (datetime.datetime.now().strftime('%Y_%m_%d_%H_%M'))) # 本次模型输出的跟文件夹
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)      # 若上方文件夹不存在，创建
        self.save_cfg() # 储存本次训练配置文件信息
        self.variable_to_restore = tf.global_variables() # 全局变量
        self.saver = tf.train.Saver(self.variable_to_restore, max_to_keep=None) #用于保存模型
        self.ckpt_file = os.path.join(self.output_dir,'ECG') # 存放模型的文件夹
        self.summary_op = tf.summary.merge_all() # 可视化操作
        self.writer = tf.summary.FileWriter(self.output_dir)
        self.global_step = tf.train.create_global_step() # 全局计数器
        self.learning_rate = tf.train.exponential_decay(
            self.initial_learning_rate, self.global_step, self.decay_steps,
            self.decay_rate, self.staircase, name='learning_rate') # 生成学习率，采用了衰退学习率
        self.optimizer = tf.train.GradientDescentOptimizer(
            learning_rate=self.learning_rate)                  # 优化器
        self.train_op = slim.learning.create_train_op(
            self.net.loss, self.optimizer, global_step=self.global_step) #训练op
        gpu_options = tf.GPUOptions()
        gpu_config = tf.ConfigProto(gpu_options=gpu_options)   # 与gpu分配相关
        self.sess = tf.Session(config=gpu_config)
        self.sess.run(tf.global_variables_initializer()) # 变量初始化
        self.loger = Loger(self.output_dir)

        if self.weight_file is not None: # 加载迁移模型
            print("从"+self.weight_file+"加载模型")
            self.saver.restore(self.sess,self.weight_file)
        self.writer.add_graph(self.sess.graph)

    """
        保存本次训练的配置信息
    """
    def save_cfg(self):
        print("保存配置信息到: "+ os.path.join(self.output_dir, 'config.txt').__str__().replace('\\','/'))
        with open(os.path.join(self.output_dir, 'config.txt'), 'w') as f:
            cfg_dict = config.__dict__
            for key in sorted(cfg_dict.keys()):
                if key[0].isupper():
                    cfg_str = '{}: {}\n'.format(key, cfg_dict[key])
                    f.write(cfg_str)
    """
        调用该函数进行训练
    """
    def train(self):
        train_timer = Timer() # 训练时间分析
        for step in range(1,self.max_iter + 1):   # 批次循环
            x_train , y_train  = self.data.get_batch() # 获取一个batch
            feed_dict = {    # 数据投放关系
                self.net.input: x_train,
                self.net.labels: y_train
            }
            if step % self.summary_iter == 0:
                # print("进行评估")
                if step % (self.summary_iter * 10) == 0: # 可修改计算loss的频率
                    # print("计算loss")
                    # 计算正确率
                    train_timer.tic()
                    loss, _ = self.sess.run(
                        [ self.net.loss, self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                    log_str = """{},step: {}, Learing rate {},Loss: {:5.3f}\n速度: {:.3f} s/iter,预计还需要： {}""".format(
                        datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                        int(step),
                        round(self.learning_rate.eval(session=self.sess), 6),
                        loss,
                        train_timer.average_time,
                        train_timer.remain(step,self.max_iter)
                    )
                    self.loger.write(log_str)
                    print(log_str)
                else:
                    # print("输出日志")
                    train_timer.tic()
                    # summary_str, _ = self.sess.run(
                    #     [self.summary_op, self.train_op],
                    #     feed_dict=feed_dict)
                    _ = self.sess.run(
                        [self.train_op],
                        feed_dict=feed_dict)
                    train_timer.toc()
                # self.writer.add_summary(summary_str, step)
            else:
                # print("单纯训练")
                train_timer.tic()
                self.sess.run(self.train_op, feed_dict=feed_dict)
                train_timer.toc()

            if step % self.save_iter == 0:
                print("{} 保存模型 : {}".format(
                    datetime.datetime.now().strftime('%m-%d %H:%M:%S'),
                    self.output_dir
                ))
                self.saver.save(self.sess,self.ckpt_file,global_step=self.global_step)
        self.loger.save()



if __name__ == '__main__':


    # tf.device('/GPU')
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    data = DATA(False)
    net = SimpleModel()
    solver = Solver(net,data)
    solver.train()
