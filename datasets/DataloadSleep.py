## 包用于对eeg_eog_5c策略的sleep-edf-78数据进行cross-validation的每一折数据进行制作
import os
import scipy.io as sio
import numpy as np
from torch.utils import data
import random
    
def load_datalist(original_path, subject_list):
    filelist = os.listdir(original_path)    # 读取路径下所有文件，保存在filelist
    dataset_list = {'data':[],'label':[]}   # 初始化为一个字典,保存数据文件名和相应标签
    # -----------------------------------------------------------
    for i in subject_list:
        data_route = os.path.join(original_path,filelist[i])
        for item in os.listdir(data_route) :   # 读取当前被试文件夹中的每一个文件名                 
            if 'C1' in item:   # 如果是类别1
                dataset_list['data'].append("{}/{}".format(data_route, item))   # 将文件名为S*_01的文件名加入data中
                dataset_list['label'].append(0)     # 标签为0，表示类别0
            elif 'C2' in item:
                dataset_list['data'].append("{}/{}".format(data_route, item))
                dataset_list['label'].append(1)
            elif 'C3' in item:
                dataset_list['data'].append("{}/{}".format(data_route, item))
                dataset_list['label'].append(2)
            elif 'C4' in item:
                dataset_list['data'].append("{}/{}".format(data_route, item))
                dataset_list['label'].append(3)
            elif 'C5' in item:
                dataset_list['data'].append("{}/{}".format(data_route, item))   ##########################################
                dataset_list['label'].append(4)
            else:
                pass       # 如果都不是，则pass
    
    return dataset_list   # 返回datalist


def load_datafile(dataset_list, idx):
    # load single data and label through path and idx
    assert os.path.isfile(dataset_list['data'][idx])
    data_array = np.load(dataset_list['data'][idx])  # 读取dataset_list中对应idx位置的数据，data_nii是字典类型
    assert data_array is not None
    data_label = dataset_list['label'][idx]
    assert data_label is not None
    # return the data array and label
    return data_array, data_label


class SleepSCDataset(data.Dataset):
    def __init__(self, root_dir,subject_list):
        assert root_dir is not None
        self.root_dir = root_dir # 
        self.subject_list = subject_list
        data_list = load_datalist(self.root_dir, self.subject_list)  # load_datalist根据val_list，返回train/test的样本名称列表，root_dir为所有被试文件夹所在目录

        self.datafile_list = data_list['data']
        self.datalabel_list = data_list['label']
        assert (len(self.datalabel_list) == len(self.datafile_list))    #判断二者的长度相等
        self.dataset_list = {'data': self.datafile_list, 'label': self.datalabel_list}

    def __len__(self):
        return len(self.datafile_list)

    def __getitem__(self, idx):
        data, label = load_datafile(self.dataset_list, idx)      # 返回一个样本和一个标签
        
        # turn into (2,1,3000)
        data_trans = self._nii2tensorarray(data)
        return data_trans.astype('float32'), label 

    # 将数据转化成4维的模式
    def _nii2tensorarray(self, data):
        
        new_data = data.transpose(1,0,2)
        
        return new_data




