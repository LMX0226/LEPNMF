import torch
import torch as th
import os
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch_geometric.nn import GATv2Conv
from torch_geometric.nn import GCNConv
from torch_geometric.nn import SAGEConv
from torch_geometric.nn import JumpingKnowledge
from torch_geometric.nn import PNAConv
from torch_geometric.nn import GatedGraphConv
from sklearn.decomposition import NMF
import numpy as np
from param import *
from utils import *
from preData import *

args = parse_args()
W,H = MYNMF(args)
class Gpslayer(nn.Module):
    def __init__(self,
                 args,
                 dim_h,
                 numheads,
                 hiddensize,
                 pna_degrees=-1,  dropout=0.3,
                 ):
        super(Gpslayer, self).__init__()
        self.activation = nn.ReLU()
        self.layer_norm = nn.LayerNorm(dim_h)
        self.dropout = nn.Dropout(dropout)
        self.linear1 = nn.Linear(dim_h,dim_h*2)
        self.linear2 = nn.Linear(dim_h*2,dim_h)
        # self.pe = PE(args.head,args.dim)
        # Defaults from the paper.
        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']
        #aggregators = ['mean', 'max', 'sum']
        #scalers = ['identity']
        # deg = th.from_numpy(np.array(pna_degrees))
        # self.local_model = PNAConv(dim_h, dim_h,
        #                                  aggregators=aggregators,
        #                                  scalers=scalers,
        #                                  deg=deg,
        #                                  edge_dim=None,
        #                                  towers=1,
        #                                  pre_layers=1,
        #                                  post_layers=1,
        #                                  divide_input=False)
        # self.local_model=GCNConv(dim_h, dim_h)
        self.local_model=GATv2Conv(args.GATf, args.GATf , args.GATh, concat=False)
        # self.local_model=SAGEConv(dim_h, dim_h)
        #self.local_model = GatedGraphConv(out_channels=dim_h,num_layers=10,aggr='add',bias=True)
        self.global_model=transform(feat_size=dim_h,
                                    hidden_size=hiddensize,
                                    num_heads=numheads,
                                    dropout=0.2,
                                    attn_dropout=0.2,
                                    activation=nn.ReLU())
        self.feature_att = FeatureAttention(args)
    def forward(self,x,edge_index):

        x_local = self.local_model(x,edge_index)
        x_local = self.dropout(x_local)
        x_local = x_local+x  #residual
        x_local = self.layer_norm(x_local)

        x_global = x
        x_global = self.global_model(x_global,edge_index)
        x_global = x_global
        x_global = self.dropout(x_global)
        x_global = x_global + x  # residual
        x_global = self.layer_norm(x_global)

        x_out = self.feature_att(x_local, x_global)

        x_out = self.layer_norm(x_out)
        x_out_re = x_out
        x_out = self.ff_block(x_out)
        x_out = x_out+x_out_re
        x_out = self.layer_norm(x_out)
        x_out = x_out
        return x_out
    def ff_block(self,x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.dropout(x)
        return self.dropout(self.linear2(x))


class transform(nn.Module):
    def __init__(
            self,
            feat_size,
            hidden_size,
            num_heads,
            norm_first=False,
            dropout=0.2,
            attn_dropout=0.2,
            activation=nn.ReLU(),
                 ):
        super(transform, self).__init__()
        self.norm_first = norm_first
        self.attn = BiasAttenWithGate(
        feat_size=feat_size,#特征
        num_heads=num_heads,#注意力头
        attn_drop=attn_dropout,
    )
        self.ffn = nn.Sequential(
        nn.Linear(feat_size, hidden_size),
        activation,
        nn.Dropout(p=dropout),
        nn.Linear(hidden_size, feat_size),
        nn.Dropout(p=dropout),
    )
        self.dropout = nn.Dropout(p=dropout)
        self.attn_layer_norm = nn.LayerNorm(feat_size)
        self.ffn_layer_norm = nn.LayerNorm(feat_size)

    def forward(self, nfeat, edge_index, attn_bias=None, attn_mask=None):
        #5.31-21.30修改
        # 对输入特征值进行layernorm处理
        nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        # 使用MHA注意力机制计算新的特征表示
        nfeat = self.attn(nfeat, edge_index, attn_bias, attn_mask)
        nfeat = self.dropout(nfeat)
        # 将原始特征和新的特征表示进行残差链接
        nfeat = residual + nfeat
        # 对残差链接后的特征进行layernorm处理
        if not self.norm_first:
            nfeat = self.attn_layer_norm(nfeat)
        residual = nfeat
        if self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        # 使用FFN前馈神经网络计算新的特征表示
        nfeat = self.ffn(nfeat)
        # 将原始特征和新的特征表示进行残差链接
        nfeat = residual + nfeat
        # 对残差链接后的特征进行layernorm处理
        if not self.norm_first:
            nfeat = self.ffn_layer_norm(nfeat)
        return nfeat


class BiasAttenWithGate(nn.Module):
    def __init__(self,
                 feat_size,
                 num_heads,
                 bias=True,
                 attn_drop=0.2):
        super(BiasAttenWithGate, self).__init__()
        self.feat_size = feat_size
        self.num_heads = num_heads
        self.head_dim = feat_size // num_heads  # 每个头部的维度
        assert (
                self.head_dim * num_heads == feat_size
        ), "feat_size must be divisible by num_heads"

        self.scaling = self.head_dim ** -0.5  # 缩放因子，保持梯度稳定

        # 投影层，用于计算Q, K, V
        self.q_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.k_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.v_proj = nn.Linear(feat_size, feat_size, bias=bias)
        self.out_proj = nn.Linear(feat_size, feat_size, bias=bias)

        # 增加门控层：更新门和重置门
        self.update_gate = nn.Linear(feat_size, feat_size, bias=True)
        self.reset_gate = nn.Linear(feat_size, feat_size, bias=True)

        # Dropout层
        self.dropout = nn.Dropout(p=attn_drop)
        self.reset_parameters()

        # Layer normalization
        self.layer_norm = nn.LayerNorm(feat_size)

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.q_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.k_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.v_proj.weight, gain=2 ** -0.5)
        nn.init.xavier_uniform_(self.out_proj.weight)

        if self.out_proj.bias is not None:
            nn.init.constant_(self.out_proj.bias, 0.0)

    def forward(self, ndata, edge_index, attn_bias=None, attn_mask=None):
        # 输入特征：ndata -> (num_nodes, feat_size)，不再有batch_size维度
        q_h = self.q_proj(ndata)
        k_h = self.k_proj(ndata)
        v_h = self.v_proj(ndata)

        node, feat_size = ndata.shape

        # 门控机制
        reset_gate = th.sigmoid(self.reset_gate(ndata))  # (num_nodes, feat_size)
        reset_gate = reset_gate.reshape(node, self.num_heads, self.head_dim)

        # 重置门对q_h和k_h的影响
        q_h = (q_h.reshape(node, self.num_heads, self.head_dim) * reset_gate) * self.scaling
        k_h = (k_h.reshape(node, self.num_heads, self.head_dim) * reset_gate).permute(1, 2, 0)
        # # 重置门对q_h和k_h的影响
        # q_h = (q_h.reshape(node, self.num_heads, self.head_dim)) * self.scaling
        # k_h = (k_h.reshape(node, self.num_heads, self.head_dim)).permute(1, 2, 0)
        v_h = v_h.reshape(node, self.num_heads, self.head_dim)

        # 计算注意力权重
        attn_weights = attenweight(q_h, k_h, edge_index, n=node, num_head=self.num_heads)
        if attn_bias is not None:
            attn_weights += attn_bias

        if attn_mask is not None:
            attn_weights[attn_mask.to(th.bool)] = float("-inf")

        attn_weights = my_softmax(attn_weights).to(args.device)
        # 因为没有batch_size维度，直接进行三维的einsum操作
        attn = th.einsum('ijk,ijl->ijl', attn_weights, v_h)

        # reshape回多头注意力的输出
        attn = attn.reshape(node, self.feat_size)

        # # 更新门用于融合自注意力输出和原始输入
        update_gate = th.sigmoid(self.update_gate(ndata))  # (num_nodes, feat_size)
        attn = update_gate * attn + (1 - update_gate) * ndata  # 门控后的输出

        # 最终经过layer norm
        attn = self.layer_norm(attn)

        return attn


class PE(nn.Module):
    def __init__(self, max_degree, embedding_dim):
        super(PE, self).__init__()
        self.encoder1 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.encoder2 = nn.Embedding(
            max_degree + 1, embedding_dim, padding_idx=0
        )
        self.max_degree = max_degree

    def forward(self, g):
        in_degree = th.clamp(g.in_degrees(), min=0, max=self.max_degree)
        out_degree = th.clamp(g.out_degrees(), min=0, max=self.max_degree)
        degree_embedding = self.encoder1(in_degree) + self.encoder2(out_degree)
        return degree_embedding
class My_feature(nn.Module):
    def __init__(self , args  ,n_rna , n_dis):
        super(My_feature, self).__init__()
        self.fc1 = nn.Linear(n_rna, args.hidden)
        self.fc2 = nn.Linear(n_dis, args.hidden)
        self.pe=PE(args.head,args.hidden)
    def forward(self,data,graph):
        M_Feature = data[0]
        D_Feature = data[1]
        M_Feature = self.fc1(M_Feature)
        D_Feature = self.fc2(D_Feature)
        # 将RNA和疾病特征在行维度进行拼接
        x = torch.cat([M_Feature, D_Feature], dim=0)
        return x


class MLP(nn.Module):
    def __init__(self, args):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(args.hidden, args.hidden // 2)
        self.bn1 = nn.BatchNorm1d(args.hidden // 2)
        self.dropout1 = nn.Dropout(args.MLPDropout)
        self.fc2 = nn.Linear(args.hidden // 2, args.hidden // 4)
        self.bn2 = nn.BatchNorm1d(args.hidden // 4)
        self.dropout2 = nn.Dropout(args.MLPDropout)
        self.fc3 = nn.Linear(args.hidden // 4, args.hidden // 8)
        self.bn3 = nn.BatchNorm1d(args.hidden // 8)
        self.dropout3 = nn.Dropout(args.MLPDropout)
        self.fc4 = nn.Linear(args.hidden // 8, 1)

    def forward(self, x):
        x = F.tanh(self.bn1(self.fc1(x)))
        x = self.dropout1(x)
        x = F.tanh(self.bn2(self.fc2(x)))
        x = self.dropout2(x)
        x = F.tanh(self.bn3(self.fc3(x)))
        x = self.dropout3(x)
        x = self.fc4(x)
        return x

class FeatureAttention(nn.Module):
    def __init__(self , args):
        super(FeatureAttention,self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(args.dim, args.dim // 2,bias=True),
            nn.ReLU(),
            nn.Linear(args.dim//2 , 1,bias=True)
        )
    def forward(self , z1, z2):
        x=self.fc(z1)
        y=self.fc(z2)
        e_x=torch.exp(x)
        e_y=torch.exp(y)
        score_x=e_x/(e_x+e_y)
        score_y=e_y/(e_x+e_y)
        diag_x=torch.diag(torch.squeeze((score_x)))
        diag_y=torch.diag(torch.squeeze((score_y)))
        z1=torch.mm(diag_x , z1)
        z2=torch.mm(diag_y , z2)
        z = z1+z2
        return  z

class Gps_Module(nn.Module):
    def __init__(self, args, n_rna, n_dis):
        super(Gps_Module, self).__init__()

        self.MLP = MLP(args)#创建MLP对象用于边的预测
        self.n_rna = n_rna
        self.n_dis = n_dis
        self.gps = Gpslayer(args,dim_h=256,numheads=4,hiddensize = 512)

        self.rnamlp = nn.Sequential(
            nn.Linear(901,256),
            nn.ReLU(),
        )

        self.diseasemlp = nn.Sequential(
            nn.Linear(877, 256),
            nn.ReLU(),
        )
        self.layernorm = nn.LayerNorm(
            512,
        )
    def encode(self,data, edge_index):

        #x = torch.Tensor(data['miRNA_disease_feature']).to(args.device)
        rna = torch.Tensor(data['miRNA_sim']).to(args.device)
        disease = torch.Tensor(data['disease_sim']).to(args.device)

        rna = self.rnamlp(rna)
        disease = self.diseasemlp(disease)

        x = torch.cat([rna, disease], dim=0)
        x = self.gps(x,edge_index)  # 使用gps特征提取
        x = self.gps(x,edge_index)  # 使用gps特征提取

        y = torch.cat([torch.tensor(W), torch.tensor(H)], dim=0).to(args.device)
        # Concatenate and cast to bfloat16 again
        x = torch.cat([x, y], dim=1).to(torch.bfloat16)
        x = x.to(torch.float32)
        # Ensure LayerNorm input is in bfloat16
        x = self.layernorm(x)
        return x

    def decode(self, z, edge_label_index):
        # z所有节点的特征向量
        src = z[edge_label_index[0]]#获取边的起始节点特征向量
        dst = z[edge_label_index[1]]#获取边的终止节点特征向量
        res = (src * dst)#逐元素相乘
        #res = torch.cat([src , dst], dim=1)
        res = res.float()
        res = self.MLP(res)#使用MLP进行边的预测
        return res

    def forward(self,data, edge_index, edge_label_index):
        z = self.encode(data, edge_index)
        res = self.decode(z, edge_label_index)
        return res
