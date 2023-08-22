# *_*coding:utf-8 *_*

import warnings
import torch
from utils.util import tps3d ,gen_3d_std_rigid , index_points,query_ball_point
import tqdm
import numpy as np
from torch.utils.data import Dataset

warnings.filterwarnings('ignore')


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


class TPS3d_dataset(Dataset):
    def __init__(self,
                 point_size=512,
                 total_data=80000,
                 deform_level=0.3,
                 drop_num=None):

        self.deform_level = deform_level
        self.point_size=point_size
        self.drop_num=drop_num
        if drop_num is not None:
            print("drop {} points".format(drop_num))

        source=np.loadtxt("./data/curtain_0001.txt",dtype=np.float32,delimiter=",")[:point_size,:3]
        # source=np.loadtxt("./data/person_0005.txt",dtype=np.float32,delimiter=",")[:point_size,:3]
       # source=np.loadtxt("./data/bunny2048.txt",dtype=np.float32,delimiter=" ")[:point_size,:3]
        source=torch.Tensor(pc_normalize(source))
        self.source=source

        ################################
        target = []
        Theta = []
        rigid=gen_3d_std_rigid()

        assert point_size % 2 == 0 #point_size只允许偶数
        half=int(point_size/2)
        for i in tqdm.trange(total_data):
            theta=rigid+(torch.rand_like(rigid)-0.5)*deform_level
            targ=tps3d(rigid,theta,source)
            #打乱targ中点的顺序 , 避免tps变换后点存在一一对应关系
            temp=targ[:half,:]
            targ[:half,:]=targ[half:,:]
            targ[half:,:]=temp

            Theta.append(theta)
            target.append(targ)

        self.theta_list = Theta
        self.target_list = target

    def __getitem__(self, index):
        target = self.target_list[index]
        theta = self.theta_list[index]
        if self.drop_num is not None:
            return (self.drop_point(target.clone())).squeeze(), (self.drop_point(self.source.clone())).squeeze(), theta
        else:
            return target,self.source,theta

    def __len__(self):
        return len(self.theta_list)

    def drop_point(self,pc:torch.Tensor) -> torch.Tensor:
        pc = pc.unsqueeze(0)    #[1,N,3]
        rand_int = torch.randint(0, self.point_size, (1,1))
        query_points=index_points(pc,rand_int)
        ball_idx=query_ball_point(0.5, self.drop_num, pc, query_points)
        batch_idx = torch.arange(1).unsqueeze(-1).unsqueeze(-1)
        tmp=torch.clone(pc[:,:self.drop_num])
        pc[batch_idx, ball_idx] = tmp
        return pc

