# *_*coding:utf-8 *_*
import torch
from utils.util import tps3d ,gen_3d_std_rigid , exclude_query_ball
import tqdm
import numpy as np
from torch.utils.data import Dataset
import random

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc


def gen_unoise():
    return torch.rand([50, 3]) * 2 - 1


class TPS3d_dataset(Dataset):
    def __init__(self,
                 point_size=512,
                 total_data=80000,
                 deform_level=0.3,
                 drop_num=None,
                 out_liner_num=None,
                 unoise=False,
                 noise=False):
        self.disorder=True
        self.noise=noise
        self.unoise=unoise
        self.deform_level = deform_level
        self.point_size=point_size
        self.drop_num=drop_num
        self.out_liner_num=out_liner_num
        if noise:
            print("adding normal noise ")
        if unoise:
            print("adding uniform noise ")
        if drop_num is not None:
            self.disorder=False
            print("drop {} points".format(drop_num))
        if out_liner_num is not None:
            assert out_liner_num<=92
            self.ball=np.loadtxt("./data/ball.txt",dtype=np.float32)[:out_liner_num,:3]
            self.ball=torch.Tensor(self.ball)/10
            self.disorder=False
            print("outline {} points".format(out_liner_num))

        source=np.loadtxt("./data/curtain_0001.txt",dtype=np.float32,delimiter=",")[:point_size,:3]
        # source=np.loadtxt("./data/person_0005.txt",dtype=np.float32,delimiter=",")[:point_size,:3]
       # source=np.loadtxt("./data/bunny2048.txt",dtype=np.float32,delimiter=" ")[:point_size,:3]
        source=torch.Tensor(pc_normalize(source))
        self.source=source

        ################################
        target = []
        Theta = []
        rigid=gen_3d_std_rigid()

        assert point_size % 2 == 0  #point_size只允许偶数
        half=int(point_size/2)
        for i in tqdm.trange(total_data):
            theta=rigid+(torch.rand_like(rigid)-0.5)*deform_level
            targ=tps3d(rigid,theta,source)

            #打乱targ中点的顺序 , 避免tps变换后点存在一一对应关系,但如果需要丢失点就不打乱
            if self.disorder:
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
            target,source=self.drop_point(target.clone(),self.source.clone())
            return target,source, theta
        elif self.out_liner_num is not None:
            target,source=self.outline(target.clone(),self.source.clone())
            return target,source, theta
        elif self.unoise:
            target=torch.cat([target,target[:50,]],dim=0)
            source=torch.cat([self.source,gen_unoise()],dim=0)
            return target,source,theta
        elif self.noise:
            target=target+torch.randn(target.size())*0.03
            source=self.source+torch.randn(self.source.size())*0.03
            return target,source,theta
        else:
            return target,self.source,theta

    def __len__(self):
        return len(self.theta_list)

    def drop_point(self,pc:torch.Tensor,pc2:torch.Tensor)  :
        pc,pc2 = pc.unsqueeze(0),pc2.unsqueeze(0)    #[1,N,3]
        rand_int = random.randint(0,self.point_size-1)             # 选一个点出来
        rand_int2 = random.randint(0,self.point_size-1)             # 选一个点出来
        query_points,query_points2=pc[:,rand_int:rand_int+1,:],pc2[:,rand_int2:rand_int2+1,:]
        pc,pc2=exclude_query_ball(self.drop_num,pc,query_points),exclude_query_ball(self.drop_num,pc2,query_points2)
        return pc.squeeze(0),pc2.squeeze(0)

    def outline(self, target:torch.Tensor,source:torch.Tensor):
        x,y,z = random.randint(-500, 500),random.randint(-500, 500),random.randint(200, 500)
        x,y,z=x/1000.,y/1000.,z/1000.
        shift=self.ball+torch.Tensor([x,y,z])
        target=torch.cat([target,shift],0)
        source=torch.cat([source,source[:self.out_liner_num,:]],0)
        return target,source

