# *_*coding:utf-8 *_*

import warnings
import numpy as np
import torch
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class cloth_dataset(Dataset):
    def __init__(self,
                 point_size=512,
                 sample_size=500,
                 use_rgb=False,
                 train:bool=True):

        dataset=np.load("data/hoody_Ll_front.npy")

        pair=[]

        total_size=dataset.shape[0]
        if sample_size>total_size:
            sample_size=total_size

        if train:
            total_data=dataset[:sample_size,...]
        else:
            total_data=dataset[0-sample_size:,...]

        if not use_rgb:
            total_data=total_data[...,:3]

        total_data=total_data[:,:point_size,:]
        for i in range(sample_size):
            total_data[i]=pc_normalize(total_data[i])

        for idx in range(sample_size):
            for idx2 in range(idx+1,sample_size):
                pair.append([total_data[idx],total_data[idx2]])

        self.pair=np.array(pair)

    def __getitem__(self, index):
        t=torch.tensor(self.pair[index])
        return t[0],t[1],t[1]       #为了保持各个数据集的读取方式一致所以加了第三个数据

    def __len__(self):
        return len(self.pair)