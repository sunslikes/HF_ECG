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
        self.loss_weight = self.get_weight() # 根据训练集计算出各个类别的比例，向量各个元素和为1




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
            if self.train_point + self.batch_size + self.batch_size >= self.train_index.__len__():
                self.echo += 1
                # self.train_point += self.batch_size
                # p1 = self.train_point
                # hx = self.read_data(self.train_index[p1:(p1+self.batch_size)%self.train_index.__len__()],
                #                   self.train_path)
                # ty = self.read_data(self.train_index[0:self.batch_size-(self.train_index.__len__()-p1+1)],
                #                   self.train_path)
                # x  = np.append(hx,ty)
                # hy = self.train_label[p1:(p1+self.batch_size)%self.train_label.__len__()]
                # ty = self.train_label[self.train_index[0:self.batch_size-(self.train_index.__len__()-p1+1)]]
                # y = np.append(hy,ty)
                # return np.array(x),np.array(y).tolist(y)
                self.train_point = -self.batch_size # 指针回零
                self.train_index,self.train_label = self.shuffle(self.train_index,self.train_label) # 重新打乱数组
            self.train_point += self.batch_size
            p = self.train_point
            x = self.read_data(self.train_index[p:(p+self.batch_size)%self.train_index.__len__()],
                                  self.train_path)
            y = self.train_label[p:(p+self.batch_size)%self.train_label.__len__()]
            return np.array(x),np.array(y).tolist()
        if type == 'test':
            if self.test_point + self.batch_size + self.batch_size >= self.test_index.__len__():
                self.test_point = -self.batch_size # 指针回零
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
    def get_weight(self):
        count_array = [0 for _ in self.class_table] # 初始化为0
        f = open(self.label_file,'r',encoding='utf-8')
        while(True):
            line = f.readline().replace('\n','')
            if line.__len__() < 1:
                break
            items = line.split('\t')
            for item in items[:50]:
                for i in range(len(self.class_table)):
                    if self.class_table[i] == item: # 查表找到
                        count_array[i] += 1
        # print(count_array)
        count_array = np.array(count_array,dtype=np.float)
        sum = np.sum(count_array)
        # print(count_array)
        # print(sum)
        weight = np.divide(count_array,sum) # 正相关weight
        weight1 = np.divide(1,np.log(count_array)) # 负相关weight
        # print(weight.tolist(),'11111')
        # print(weight1.tolist())
        # print(np.sum(weight))
        return weight1








if __name__ == '__main__':
    data = DATA()
    data.get_weight()
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
    # data.train_point = 22000
    # data.batch_size = 1000
    # print(data.get_batch('train')[0].shape)
    # print(data.echo)
    # print(data.train_point)
    # print(data.get_batch('train')[0].shape)
    # print(data.echo)
    # print(data.train_point)
    # print(data.get_batch('train')[0].shape)
    # print(data.get_batch('train')[0].shape)
    # print(data.train_index[0])
    # print(data.echo)
    # print(data.train_index[0])
    # print(data.train_label[0])
    # print(data.get_batch('train')[0])
    # print(data.echo)
    # print(data.train_index[0])
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
