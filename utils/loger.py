import os

class Loger:
    def __init__(self,path):
        self.path = os.path.join(path,'logs.txt')
        self.file = open(self.path,'w',encoding='utf-8')
        if not os.path.exists(self.path):
            os.makedirs(self.path)

    def write(self,content):
        self.file.writelines(content)
    def save(self):
        print("日志保存在" + str(self.path))
        self.file.close()