import os
import numpy as np
from utils import config as cfg
import re
class DATA():
    def __init__(self):
        self.train_path = os.path.join(cfg.DATA_PATH,cfg.TRAIN_PATH) # 训练集目录
        self.test_path = os.path.join(cfg.DATA_PATH,cfg.TEST_PATH) # 测试集目录
        self.label_file = os.path.join(cfg.DATA_PATH,cfg.LABEL_PATH) # 训练标注
        self.test_label_file = os.path.join(cfg.DATA_PATH,cfg.TEST_LABEL) # 测试答案
        self.class_table_file = os.path.join(cfg.DATA_PATH,cfg.ARRYTHMIA_CLASSES_PATH) # 分类文件路径
        self.batch_size = cfg.BATCH_SIZE
        self.class_table = self.get_class_table()  # 读取类别文件，储存成列表
        self.train_label,self.train_index = self.get_label_and_index(self.label_file)
        self.train_index,self.train_label = self.shuffle(self.train_index,self.train_label) #随机打乱
        self.test_label,self.test_index = self.get_label_and_index(self.test_label_file)
        self.train_point = -self.batch_size # 训练指针
        self.test_point = -self.batch_size # 测试指针
        self.echo = 0 # 训练集循环数




    '''
     获取类别列表
    '''
    def get_class_table(self):
        f = open(self.class_table_file,"r",encoding='utf-8')
        li = []
        while(True):
            line = f.readline().replace('\n','')
            if line.__len__() < 1:
                break
            li.append(line)
        return li
    '''
    获取标注，以及训练集的索引
    返回标注（numpy数组）与标签
    '''
    def get_label_and_index(self,label_path):
        f = open(label_path,'r',encoding='utf-8')
        label = []
        index = []
        while(True):
            line = f.readline().replace('\n','')
            if line.__len__() < 1:
                break
            a = line.split('\t') # a是把一行切割出来的元素数组
            index.append(a[0]) # 索引记录起来
            la = np.zeros(self.class_table.__len__())  #[0,0,0,...0,0,0]
            for _ in a:
                for i in range(self.class_table.__len__()):
                    if _ == self.class_table[i]: # 若出现标注
                        la[i] = 1
                        break
            label.append(la)
        return label,index
    '''
    根据索引列表返回一个batch
    '''
    def read_data(self,index_list,header):
        rst = []
        for index in index_list:
            # print('打开文件' + os.path.join(header,index))
            f = open(os.path.join(header,index)) # 打开数据集
            l = []
            f.readline() #
            while(True):
                line = f.readline().replace('\n','')
                if line.__len__() < 1:
                    break
                a = line.split(' ')
                l.append(a) # 加入
            nl = np.array(l) # 转化为np
            rst.append(nl)
        return np.array(rst,dtype='int32')
    '''
    获取一个batch
    '''
    def get_batch(self,type):
        if type == 'train':
            if self.train_point + self.batch_size >= self.train_index.__len__():
                self.echo += 1
            self.train_point += self.batch_size
            p = self.train_point
            x = self.read_data(self.train_index[p:(p+self.batch_size)%self.train_index.__len__()],
                                  self.train_path)
            y = self.train_label[p:(p+self.batch_size)%self.train_label.__len__()]
            return np.array(x),np.array(y).tolist()
        if type == 'test':
            self.test_point += self.batch_size
            p = self.test_point
            x = self.read_data(self.test_index[p:(p+self.batch_size)%self.test_index.__len__()],
                                  self.test_path)
            y = self.test_label[p:(p+self.batch_size)%self.test_label.__len__()]
            return np.array(x),np.array(y).tolist()
        print("请输入 [train] 或 [test] 作为参数")
    def shuffle(self,index,label):
        a = np.array([index,label])
        a = a.transpose() # 转置
        np.random.shuffle(a)
        a = a.transpose() # 转置回来
        return a[0],a[1]








if __name__ == '__main__':
    data = DATA()
    # index = data.train_index[0:5]
    # label = data.train_label[0:5]
    # a = np.array([index,label])
    # print(a.shape)
    # print(a)
    # a = a.transpose()
    # np.random.shuffle(a)
    # print(a.shape)
    # print(a)
    # a = a.transpose()
    # print(a)

    print(data.train_index[0])
    print(data.train_label[0])
    print(data.get_batch('train')[0])
    # x,y = data.get_batch('train')
    # xt,yt = data.get_batch('test')
    # print(xt[0:5])
    # print('----------------------------------------------')
    # print(yt[0:5])
    # print(y.dtype)
    # print(xt[0])
    # print(data.read_data(data.train_index[0:5],data.train_path).shape)
    # print(x.transpose(0,2,1))
    # print(x.dtype,"11111")
    # print(y)
    # print(data.test_label)
