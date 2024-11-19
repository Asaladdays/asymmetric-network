
import torch
import torch.nn as nn
import torch.nn.functional as F
from pointnet_util import farthest_point_sample, index_points, square_distance, PointNetFeaturePropagation




def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)

    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx

class Local_op(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        b, n, s, d = x.size()  # torch.Size([32, 512, 32, 128])     output [32,512,128]
        x = x.permute(0, 1, 3, 2)
        x = x.reshape(-1, d, s)
        batch_size, _, N = x.size()
        x = self.relu(self.bn1(self.conv1(x)))  # B, D, N
        x = self.relu(self.bn2(self.conv2(x)))  # B, D, N
        x = torch.max(x, 2)[0]
        x = x.view(batch_size, -1)
        x = x.reshape(b, n, -1).permute(0, 2, 1)
        # print(x.shape)
        return x

class LocalGrouper(nn.Module):
    def __init__(self, channel, groups, kneighbors, use_xyz=True, normalize="center", **kwargs):
        """
        Give xyz[b,p,3] and fea[b,p,d], return new_xyz[b,g,3] and new_fea[b,g,k,d]
        :param groups: groups number
        :param kneighbors: k-nerighbors
        :param kwargs: others
        """
        super(LocalGrouper, self).__init__()
        self.groups = groups
        self.kneighbors = kneighbors
        self.use_xyz = use_xyz
        if normalize is not None:
            self.normalize = normalize.lower()
        else:
            self.normalize = None
        if self.normalize not in ["center", "anchor"]:
            print(f"Unrecognized normalize parameter (self.normalize), set to None. Should be one of [center, anchor].")
            self.normalize = None
        if self.normalize is not None:
            add_channel = 3 if self.use_xyz else 0
            self.affine_alpha = nn.Parameter(torch.ones([1, 1, 1, channel + add_channel]))
            self.affine_beta = nn.Parameter(torch.zeros([1, 1, 1, channel + add_channel]))

    def forward(self, xyz, points):
        B, N, C = xyz.shape
        S = self.groups
        xyz = xyz.contiguous()  # xyz [btach, points, xyz]
        # print(self.affine_alpha.shape)
        # print(self.affine_beta.shape)
        # fps_idx = torch.multinomial(torch.linspace(0, N - 1, steps=N).repeat(B, 1).to(xyz.device), num_samples=self.groups, replacement=False).long()
        # fps_idx = farthest_point_sample(xyz, self.groups).long()
        fps_idx = farthest_point_sample(xyz, self.groups).long()  # [B, npoint]
        new_xyz = index_points(xyz, fps_idx)  # [B, npoint, 3]
        new_points = index_points(points, fps_idx)  # [B, npoint, d]

        idx = knn_point(self.kneighbors, xyz, new_xyz)
        # idx = query_ball_point(radius, nsample, xyz, new_xyz)
        grouped_xyz = index_points(xyz, idx)  # [B, npoint, k, 3]
        grouped_points = index_points(points, idx)  # [B, npoint, k, d]
        if self.use_xyz:
            grouped_points = torch.cat([grouped_points, grouped_xyz], dim=-1)  # [B, npoint, k, d+3]
        if self.normalize is not None:
            if self.normalize == "center":
                mean = torch.mean(grouped_points, dim=2, keepdim=True)
            if self.normalize == "anchor":
                mean = torch.cat([new_points, new_xyz], dim=-1) if self.use_xyz else new_points
                mean = mean.unsqueeze(dim=-2)  # [B, npoint, 1, d+3]
            std = torch.std((grouped_points - mean).reshape(B, -1), dim=-1, keepdim=True).unsqueeze(dim=-1).unsqueeze(
                dim=-1)
            grouped_points = (grouped_points - mean) / (std + 1e-5)

            grouped_points = self.affine_alpha * grouped_points + self.affine_beta
            # print(grouped_points.shape)

        new_points = torch.cat([grouped_points, new_points.view(B, S, 1, -1).repeat(1, 1, self.kneighbors, 1)], dim=-1)
        return new_xyz, new_points

def knn(x, k):
    inner = -2 * torch.matmul(x.transpose(2, 1), x)
    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)

    idx = pairwise_distance.topk(k=k, dim=-1)[1]  # (batch_size, num_points, k)
    return idx


def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points)
    if idx is None:
        idx = knn(x, k=k)  # (batch_size, num_points, k)
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size,device=device).view(-1, 1, 1) * num_points
    idx_base = idx_base.type(torch.cuda.LongTensor)
    idx = idx.type(torch.cuda.LongTensor)
    idx = idx + idx_base
    idx = idx.view(-1)

    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()  # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size * num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims)
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)

    feature = torch.cat((feature - x, x), dim=3).permute(0, 3, 1, 2)

    return feature


class CAA_Module(nn.Module):


    def __init__(self, in_dim):
        super(CAA_Module, self).__init__()

        self.bn1 = nn.BatchNorm1d(in_dim // 8)
        self.bn2 = nn.BatchNorm1d(in_dim // 8)
        self.bn3 = nn.BatchNorm1d(in_dim)

        self.query_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False),
                                        self.bn1,
                                        nn.ReLU())
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1, bias=False),
                                      self.bn2,
                                      nn.ReLU())
        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_dim, out_channels=in_dim, kernel_size=1, bias=False),
                                        self.bn3,
                                        nn.ReLU())

        self.alpha = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):


        x_hat = x.permute(0, 2, 1)
        proj_query = self.query_conv(x_hat)
        proj_key = self.key_conv(x_hat).permute(0, 2, 1)
        similarity_mat = torch.bmm(proj_key, proj_query)


        affinity_mat = torch.max(similarity_mat, -1, keepdim=True)[0].expand_as(similarity_mat) - similarity_mat
        affinity_mat = self.softmax(affinity_mat)
        proj_value = self.value_conv(x_hat)

        out = torch.bmm(proj_value,affinity_mat)

        out = self.alpha * out + x_hat
        return out


class ABEM_Module(nn.Module):


    def __init__(self, in_dim, out_dim,N, k):
        super(ABEM_Module, self).__init__()

        self.k = k
        self.bn1 = nn.BatchNorm2d(out_dim)
        self.conv1 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn2 = nn.BatchNorm2d(in_dim)
        self.conv2 = nn.Sequential(nn.Conv2d(out_dim, in_dim, kernel_size=[1, self.k], bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.bn3 = nn.BatchNorm2d(out_dim)
        self.conv3 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa1 = CAA_Module(N)

        self.bn4 = nn.BatchNorm2d(out_dim)
        self.conv4 = nn.Sequential(nn.Conv2d(in_dim * 2, out_dim, kernel_size=[1, self.k], bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))

        self.caa2 = CAA_Module(N)

    def forward(self, x):

        x1 = x
        input_edge = get_graph_feature(x, k=self.k)
        x = self.conv1(input_edge)
        x2 = x

        x = self.conv2(x)
        x = torch.squeeze(x, -1)
        x3 = x

        delta = x3 - x1

        x = get_graph_feature(delta, k=self.k)
        x = self.conv3(x)
        x4 = x

        x = x2 + x4
        x5 = x.max(dim=-1, keepdim=False)[0]
        x5 = self.caa1(x5)


        x_local = self.conv4(input_edge)
        x_local = torch.squeeze(x_local, -1)
        x_local = self.caa2(x_local)  # B,out_dim,N

        x = x5 * x_local
        return x



class PointCloudTransformerSetAbstraction(nn.Module):
    def __init__(self,channel=32,groups=512,kneighbors=32,k=16,use_xyz=False,normalize="anchor"):
        super(PointCloudTransformerSetAbstraction,self).__init__()
        self.channel=channel
        self.groups=groups
        self.kneighbors=kneighbors
        self.use_xyz=use_xyz
        self.normalize=normalize
        self.K = k
        self.dim=2*channel
        self.LG=LocalGrouper(channel=self.channel,groups=self.groups,kneighbors=self.kneighbors,use_xyz=self.use_xyz, normalize=self.normalize)
        if self.use_xyz:
           self.LO=Local_op(in_channels=self.channel*2+3,out_channels=self.channel*2)
        else:
           self.LO = Local_op(in_channels=self.channel * 2 , out_channels=self.channel*2)

        self.SA=ABEM_Module(in_dim=self.dim,out_dim=self.dim,N=self.groups, k=self.K)


    def forward(self,xyz,points):

        new_xyz,new_points=self.LG(xyz,points)
        gather_points=self.LO(new_points)
        SA_points=self.SA(gather_points)

        return  new_xyz, SA_points

class PosE_Initial(nn.Module):
    def __init__(self, in_dim, out_dim, alpha, beta):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.alpha, self.beta = alpha, beta

    def forward(self, xyz):
        device = torch.device('cuda')
        B, _, N = xyz.shape
        feat_dim = self.out_dim // (self.in_dim*2)
        feat_range = torch.arange(feat_dim,device=device).float()
        dim_embed = torch.pow(self.alpha, feat_range / feat_dim)
        div_embed = torch.div(self.beta * xyz.unsqueeze(-1), dim_embed)


        sin_embed = torch.sin(div_embed)
        cos_embed = torch.cos(div_embed)
        position_embed = torch.stack([sin_embed, cos_embed], dim=4).flatten(3)
        position_embed = position_embed.permute(0, 1, 3, 2).reshape(B, self.out_dim, N)

        return position_embed

class PointTransformerSeg(nn.Module):
    def __init__(self, k=6, input_dim=9, channels=[72, 96, 128],N=2048,kneighbors=32,use_xyz=True,normalize="anchor"):
        super(PointTransformerSeg, self).__init__()
        self.k = k
        self.channel=channels
        self.N=N
        self.groups=[N/4,N/16,N/64]
        self.kneighbors=kneighbors
        self.use_xyz=use_xyz
        self.normalize=normalize
        d_points = input_dim
        self.conv1 = nn.Conv1d(d_points, int(channels[0]/2), kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(int(channels[0]/2), int(channels[0]), kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(int(channels[0]/2))
        self.bn2 = nn.BatchNorm1d(channels[0])

        self.raw_point_embedding=PosE_Initial(input_dim,channels[0],alpha=50,beta=500)

        self.PCTSA1 = PointCloudTransformerSetAbstraction(channel=self.channel[0], groups=int(self.groups[0]),
                                                          kneighbors=self.kneighbors, use_xyz=self.use_xyz, normalize=self.normalize)
        self.PCTSA2 = PointCloudTransformerSetAbstraction(channel=self.channel[1], groups=int(self.groups[1]),
                                                          kneighbors=self.kneighbors, use_xyz=self.use_xyz,normalize=self.normalize)
        self.PCTSA3 = PointCloudTransformerSetAbstraction(channel=self.channel[2], groups=int(self.groups[2]),
                                                          kneighbors=self.kneighbors, use_xyz=self.use_xyz,normalize=self.normalize)


        self.fp3=PointNetFeaturePropagation(864,[512,256])
        self.fp2=PointNetFeaturePropagation(400,[256,128])
        self.fp1=PointNetFeaturePropagation(200,[128,64])



        self.convs1 = nn.Conv1d(64, 32, 1)
        self.dp1 = nn.Dropout(0.5)
        self.bns1 = nn.BatchNorm1d(32)

        self.convs2 = nn.Conv1d(32, self.k, 1)
        # self.bns2 = nn.BatchNorm1d(256)

        self.relu = nn.ReLU()

    def forward(self, x):
        l0_xyz = x[..., :3]
        x = x.permute(0, 2, 1)
        batch_size, _, N = x.size()
        x=self.raw_point_embedding(x)
        l0_points= x.permute(0,2,1)
        l1_xyz,l1_points=self.PCTSA1(l0_xyz,l0_points)
        l2_xyz, l2_points = self.PCTSA2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.PCTSA3(l2_xyz, l2_points)


        L2_points=self.fp3(l2_xyz,l3_xyz,l2_points,l3_points)
        L1_points=self.fp2(l1_xyz,l2_xyz,l1_points,L2_points)
        L0_points=self.fp1(l0_xyz,l1_xyz,l0_points,L1_points)

        L0_points=L0_points.permute(0,2,1)

        x=self.dp1(self.relu((self.bns1(self.convs1(L0_points)))))
        x=self.convs2(x)
        x=F.log_softmax(x,dim=1)
        x=x.permute(0,2,1)
        return x


