import os
import uuid

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class PointNetEncoder(nn.Module):
    def __init__(self, channel=3, d_model=512):
        super(PointNetEncoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(channel, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, d_model, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(d_model)

    def forward(self, x):
        """
        :param x: torch tensor [B, D, N]
        :return: torch tensor [B, d_model, N]
        """
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.leaky_relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        return x

def index_points(points, idx):
    device = points.device
    B = points.shape[0]
    #这是从pointnet2里面搬出来的,那里面的索引可能有三个维度,也可能有两个维度,所以加了一个判断
    #取索引的原理就是利用pytorch的广播机制,生成对应维度的索引
    if len(idx.shape) == 2:
        batch_idx = torch.arange(B).unsqueeze(-1).to(device)
        new_points = points[batch_idx, idx]
        return new_points
    else:
        batch_idx = torch.arange(B).unsqueeze(-1).to(device).unsqueeze(-1)
        new_points = points[batch_idx, idx]
        return new_points

def farthest_point_sample(xyz, npoint):
    #一个假的FPS实现
    device = xyz.device
    B, N, C = xyz.shape
    return torch.arange(npoint,device=device).repeat(B,1)


def square_distance(src, dst):
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist

def exclude_query_ball(K:int,xyz:torch.Tensor,query_point:torch.Tensor):
    '''

    :param K:
    :param xyz: 整个点云 [1,N,3]
    :param query_point: 一个点. 在这个点周围做K近邻查询 [1,1,3]
    :return: idx
    '''
    _,N,_=xyz.size()
    square_dist = square_distance(xyz, query_point)      #[1,N,1]
    indices = torch.topk(square_dist, N-K, 1, False, False)[1].squeeze(-1)
    new_xyz=index_points(xyz,indices)
    return new_xyz


def query_ball_point(radius, nsample, xyz, new_xyz):
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

class PointNetSetAbstractionMsg(nn.Module):
    def __init__(self, npoint, radius_list, nsample_list, in_channel, mlp_list):
        super(PointNetSetAbstractionMsg, self).__init__()
        self.npoint = npoint
        self.radius_list = radius_list
        self.nsample_list = nsample_list
        self.conv_blocks = nn.ModuleList()
        self.bn_blocks = nn.ModuleList()
        for i in range(len(mlp_list)):
            convs = nn.ModuleList()
            bns = nn.ModuleList()
            last_channel = in_channel + 3
            for out_channel in mlp_list[i]:
                convs.append(nn.Conv2d(last_channel, out_channel, 1))
                bns.append(nn.BatchNorm2d(out_channel))
                last_channel = out_channel
            self.conv_blocks.append(convs)
            self.bn_blocks.append(bns)


    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, C, N]
            points: input points data, [B, D, N]
        Return:
            new_xyz: sampled points position data, [B, C, S]
            new_points_concat: sample points feature data, [B, D', S]
        """
        xyz = xyz.permute(0, 2, 1)
        if points is not None:
            points = points.permute(0, 2, 1)
        B, N, C = xyz.shape
        S = self.npoint
        # 这里不进行下采样 , 所以fps做一个假的实现
        new_xyz = index_points(xyz, farthest_point_sample(xyz, S))
        new_points_list = []
        for i, radius in enumerate(self.radius_list):
            K = self.nsample_list[i]
            group_idx = query_ball_point(radius, K, xyz, new_xyz)
            grouped_xyz = index_points(xyz, group_idx)
            grouped_xyz -= new_xyz.view(B, S, 1, C)
            if points is not None:
                grouped_points = index_points(points, group_idx)
                grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)
            else:
                grouped_points = grouped_xyz

            grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, D, K, S]
            for j in range(len(self.conv_blocks[i])):
                conv = self.conv_blocks[i][j]
                bn = self.bn_blocks[i][j]
                grouped_points =  F.relu(bn(conv(grouped_points)))
            new_points = torch.max(grouped_points, 2)[0]  # [B, D', S]
            new_points_list.append(new_points)

        new_xyz = new_xyz.permute(0, 2, 1)
        new_points_concat = torch.cat(new_points_list, dim=1)
        return new_xyz, new_points_concat

class PointNetDecoder(nn.Module):
    def __init__(self, d_model=512):
        super(PointNetDecoder, self).__init__()
        self.conv1 = torch.nn.Conv1d(d_model, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 32, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        """
        :param x: torch tensor [B, D, N]
        :return: torch tensor [B, d_model, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x

class DispGenerator(nn.Module):
    def __init__(self, d_model=512):
        super(DispGenerator, self).__init__()
        self.conv1 = torch.nn.Conv1d(d_model, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 32, 1)
        self.conv3 = torch.nn.Conv1d(32, 3, 1)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(32)

    def forward(self, x):
        """
        :param x: torch tensor [B, D, N]
        :return: torch tensor [B, d_model, N]
        """
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.conv3(x)
        return x

class STNkd(nn.Module):
    def __init__(self, k=64):
        super(STNkd, self).__init__()
        self.conv1 = torch.nn.Conv1d(k, 64, 1)
        self.conv2 = torch.nn.Conv1d(64, 128, 1)
        self.conv3 = torch.nn.Conv1d(128, 1024, 1)
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, k * k)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = Variable(torch.from_numpy(np.eye(self.k).flatten().astype(np.float32))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


def chamfer_loss(x, y, ps=91):
    """
    compute chamfer_loss between two point cloud
    :param x: tensor. [B,C,N]
    :param y: tensor. [B,C,N]
    :param ps:
    :return: torch.float
    """

    A = x.permute(0, 2, 1)
    B = y.permute(0, 2, 1)
    r = torch.sum(A * A, dim=2)
    r = r.unsqueeze(-1)
    r1 = torch.sum(B * B, dim=2)
    r1 = r1.unsqueeze(-1)

    temp1 = r.repeat(1, 1, ps)
    temp2 = -2 * torch.bmm(A, B.permute(0, 2, 1))
    temp3 = r1.permute(0, 2, 1).repeat(1, ps, 1)
    t = temp1 + temp2 + temp3
    d1, _ = t.min(dim=1)
    d2, _ = t.min(dim=2)
    ls = (d1 + d2) / 2
    return ls.mean()


def gaussian_mix_loss(x, y, var=1, ps=91, w=0, sigma=20):
    """

    :param x: tensor. [B,C,N]
    :param y: tensor. [B,C,N]
    :param var:
    :param ps:
    :param w:
    :param sigma:
    :return: torch.float
    """
    # center is B
    A = x.permute(0, 2, 1)
    B = y.permute(0, 2, 1)
    bs = A.shape[0]
    ps = A.shape[1]
    A = (A.unsqueeze(2)).repeat(1, 1, ps, 1)
    B = (B.unsqueeze(1)).repeat(1, ps, 1, 1)
    sigma_inverse = ((torch.eye(2) * (1.0 / var)).unsqueeze(0).unsqueeze(0).unsqueeze(0)).repeat(
        [bs, ps, ps, 1, 1]).cuda()
    sigma_inverse = sigma * sigma_inverse
    sigma_inverse = sigma_inverse.view(-1, 2, 2)
    tmp1 = (A - B).unsqueeze(-2).view(-1, 1, 2)
    tmp = torch.bmm(tmp1, sigma_inverse)
    tmp = torch.bmm(tmp, tmp1.permute(0, 2, 1))
    tmp = tmp.view(bs, ps, ps)
    tmp = torch.exp(-0.5 * tmp)
    tmp = tmp / (2 * np.pi * var)
    tmp1 = tmp.sum(dim=-1)
    tmp2 = tmp.sum(dim=1)
    tmp1 = torch.clamp(tmp1, min=0.01)
    return (-torch.log(tmp1 / 90.0)).mean()


def PC_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    compute euclidean distance between two point cloud
    :param x: point cloud x [B,C,N]
    :param y:point cloud y [B,C,N]
    :return: [B,1]
    """
    return torch.mean((x - y) ** 2, [1, 2])


def pairwise_l2_norm2_batch(src, dst)->torch.Tensor:
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def regularizing_loss(pc_a:torch.Tensor,pc_b:torch.Tensor,pred_a:torch.Tensor,pred_b:torch.Tensor)->torch.Tensor:
    # Tensor format [B,N,C]
    displacements_A = torch.cat([pc_a, pred_b],dim=2)
    displacements_B = torch.cat([pred_a,pc_b],dim=2)

    square_dist = pairwise_l2_norm2_batch( displacements_A,   displacements_B )
    dist = torch.sqrt(square_dist)

    minRow ,_= torch.min(dist, dim=2)
    minCol ,_= torch.min(dist, dim=1)
    RegularLoss = (torch.mean(minRow) + torch.mean(minCol))/2

    return RegularLoss


def save_pc2visual(save_dir, epoch, point_set1, point_set2, warped):
    id = str(uuid.uuid4())
    save_path = os.path.join(save_dir, str(epoch) + 'epoch_' + id)
    os.mkdir(save_path)
    l = point_set1.size()[0]
    point_set1, point_set2, warped = point_set1.cpu().numpy(), point_set2.cpu().numpy(), warped.cpu().numpy()
    for i in range(l):
        np.savetxt(os.path.join(save_path, str(i) + "pc1.txt"), point_set1[i])
        np.savetxt(os.path.join(save_path, str(i) + "pc2.txt"), point_set2[i])
        np.savetxt(os.path.join(save_path, str(i) + "warped.txt"), warped[i])


import ctypes
from torch.autograd import Function

if torch.cuda.is_available():
    lib = ctypes.cdll.LoadLibrary("libmorton/encode.so")
    lib.encode.restype = ctypes.c_uint64


def z_order_encode(inputs):
    shape = list(inputs.shape)
    shape[-1] = 1
    code = np.ndarray(shape, dtype=np.uint64)
    for i in range(shape[0]):
        for j in range(shape[1]):
            x, y, z = inputs[i, j].tolist()
            code[i, j] = lib.encode(x, y, z)
    return code.astype(np.float64)


class Z_order_sorting(Function):
    @staticmethod
    def forward(ctx, xyz):
        min=torch.min(xyz,dim=1)[0].unsqueeze(1)
        data = ((xyz + min) * 256).cpu().numpy()
        data = data.astype(dtype=np.uint32)
        assert data.shape[-1] == 3
        z_order_code = torch.from_numpy(z_order_encode(data)).cuda()
        _, idx = torch.sort(z_order_code, dim=1)
        batch_idx = torch.arange(xyz.shape[0]).reshape(xyz.shape[0], 1, 1)
        return xyz[batch_idx, idx].squeeze(2) #, normal[batch_idx, idx].squeeze(2)

    @staticmethod
    def backward(ctx, grad_out):
        return ()



def tps3d(old_ctl :torch.Tensor,new_ctl:torch.Tensor,source:torch.Tensor)->torch.Tensor:
    """
    ues tps3d to warp source point cloud
    :param old_ctl: old control points coordinates  [27,C]
    :param new_ctl: new control points coordinates  [27,C]
    :param source: source point cloud need to be warpped  [N,C]
    :return: warpped point cloud
    """
    # epsilon=1e-320
    npnts,C=old_ctl.size()
    K=torch.zeros([npnts,npnts])
    for rr in range(npnts):
        for cc in range(npnts):
            K[rr,cc]=torch.sum((old_ctl[rr,:]-old_ctl[cc,:])**2)
            K[cc,rr]=K[rr,cc]

    K=torch.maximum(K,torch.zeros_like(K)+1e-32)
    K=torch.sqrt(K)
    P=torch.cat([torch.ones((npnts,1)),old_ctl],-1)
    L=torch.cat(
        [torch.cat([K,P],dim=-1),
        torch.cat([P.T,torch.zeros((4,4))],dim=-1)]
    ,dim=0)
    param=torch.pinverse(L) @ torch.cat([new_ctl,torch.zeros((4,3))],dim=0)

    pntsNum=source.size()[0]
    K=torch.zeros(pntsNum,npnts)
    gx=source[:,0].reshape(-1,1)
    gy=source[:,1].reshape(-1,1)
    gz=source[:,2].reshape(-1,1)
    # print(gx-old_ctl[0,0])    #[128,1]
    for  kk in range(npnts):
        K[:,[kk]]=((gx-old_ctl[kk,0])**2 + (gy-old_ctl[kk,1])**2 + (gz-old_ctl[kk,2])**2).float()

    K=torch.maximum(K,torch.zeros_like(K)+1e-32)
    K=torch.sqrt(K)

    P=torch.cat([torch.ones((pntsNum,1)),gx,gy,gz],dim=-1)
    L=torch.cat([K,P],dim=-1).float()
    return L @ param

def gen_3d_std_rigid():
    rigid_3d = []
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            for k in [-1, 0, 1]:
                rigid_3d.append([i, j, k])
    rigid_3d=torch.tensor(rigid_3d).reshape(27,3).float()
    return rigid_3d


def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    m = np.max(np.sqrt(np.sum(pc ** 2, axis=1)))
    pc = pc / m
    return pc

if __name__ == '__main__':

    xyz=torch.rand([4,32,3])
    torch_sum = torch.sum(xyz, dim=0)
    print(torch_sum.size())
