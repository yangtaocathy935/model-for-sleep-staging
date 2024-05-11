from datasets.DataloadSleep import SleepSCDataset
from sklearn.model_selection import KFold
import os

def split_generator(fold_num,data_rootpath,data_source,random_state):  
    if data_source == 'edf20':
        data_list = filter_sample(data_rootpath)
        kf=KFold(fold_num,shuffle=False)  # 进行fold_num折交叉验证，对被试顺序按照固定的random_state进行打乱后分割 
    else:
        data_list = os.listdir(data_rootpath)
        kf=KFold(fold_num,shuffle=True,random_state=random_state)  # 进行fold_num折交叉验证，对被试顺序按照固定的random_state进行打乱后分割 
    split_train_index = []
    split_test_index = []
    for train_index, test_index in kf.split(data_list):
        split_train_index.append(train_index)
        split_test_index.append(test_index)   # 每一折test的index为一个array,将所有折的array保存在一个列表返回
    return split_train_index,split_test_index

def filter_sample(data_root):
    data_all = os.listdir(data_root)
    data_list = []
    for i in data_all:
        if int(i[4:6])<20:
            data_list.append(i)
    return data_list

class SleepSC_Gene:
    def __init__(self,fold_num, data_rootpath,data_source,random_state=1):
        assert data_rootpath is not None 
        self.fold_num = fold_num
        self.data_rootpath = data_rootpath
        self.data_source = data_source
        self.random_state = random_state
        self.split_train, self.split_test = split_generator(self.fold_num,self.data_rootpath,self.data_source,self.random_state)

    def train_test_data(self,fold_order,data_rootpath):
        TrainList = SleepSCDataset(data_rootpath,self.split_train[fold_order]) # 训练
        TestList =  SleepSCDataset(data_rootpath,self.split_test[fold_order])  # 测试
        return TrainList, TestList
    
    

   

        
