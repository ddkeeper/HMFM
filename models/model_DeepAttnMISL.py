from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *

class DeepAttnMISL(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10, size_arg="small", dropout=0.25):
        super(DeepAttnMISL, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        self.num_clusters = num_clusters
        
        size = self.size_dict[size_arg]
        
        self.phis = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, size[1]),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(size[1], size[1]),
                nn.ReLU(),
                nn.Dropout(dropout)
            ) for _ in range(num_clusters)
        ])
        self.attention_net = nn.Sequential(
            nn.Linear(size[1], size[1]),
            nn.ReLU(),
            nn.Dropout(dropout),
            Attn_Net_Gated(L=size[1], D=size[2], dropout=dropout, n_classes=1)
        )
        self.rho = nn.Sequential(
            nn.Linear(size[1], input_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        batch_size, n1, m = x.shape
        
        h_cluster = []
        for i in range(self.num_clusters):
            h_cluster_i = self.phis[i](x.view(-1, m)).view(batch_size, n1, -1)
            h_cluster.append(h_cluster_i)
        h_cluster = torch.stack(h_cluster, dim=2)
        
        A, h = self.attention_net(h_cluster)
        #print(A.shape)
        #print(h.shape)
        
        A = A.squeeze(0).transpose(1, 2)  # 移除最后一个维度，并转置
        A = F.softmax(A, dim=-1)
        h = h.squeeze(0)  # 移除第一个维度
        #print(A.shape)
        #print(h.shape)
        h = torch.bmm(A, h)
        #print(h.shape)
        h = self.rho(h.view(-1, h.size(-1))).view(batch_size, n1, m)
        
        return h

#first DeepAttnMISL, then cross-attention
class DAMISLFusion1(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10, size_arg="small", dropout=0.25):
        super(DAMISLFusion1, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        
        self.deepattnmisl1 = DeepAttnMISL(input_dim, num_clusters, size_arg, dropout)
        self.deepattnmisl2 = DeepAttnMISL(input_dim, num_clusters, size_arg, dropout)
        # 交叉注意力层
        self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout)
        
        # 添加一个层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 添加一个前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim)
        )

    def forward(self, x1, x2):
        x1 = self.deepattnmisl1(x1)
        x2 = self.deepattnmisl2(x2)
        #print(f"x1.shape: {x1.shape}")
        #print(f"x2.shape: {x2.shape}")
        # 交叉注意力融合
        x1_t = x1.transpose(0, 1)  # (n1, batch_size, input_dim)
        x2_t = x2.transpose(0, 1)  # (n2, batch_size, input_dim)
        
        # 使用 x1 作为查询，x2 作为键和值
        attn_output, _ = self.cross_attention(x1_t, x2_t, x2_t)
        
        # 残差连接和层归一化
        fused_features = self.layer_norm(x1_t + attn_output)
        
        # 前馈网络
        fused_features = fused_features + self.feed_forward(fused_features)
        
        # 转置回原始形状
        fused_features = fused_features.transpose(0, 1)  # (batch_size, n1, input_dim)
        
        return fused_features

#first cross-attention, then DeepAttnMISL
class DAMISLFusion2(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10, size_arg="small", dropout=0.25):
        super(DAMISLFusion2, self).__init__()
        self.size_dict = {"small": [1024, 512, 256], "big": [1024, 512, 384]}
        size = self.size_dict[size_arg]
        
        # 交叉注意力层
        #self.cross_attention = nn.MultiheadAttention(embed_dim=input_dim, num_heads=4, dropout=dropout)
        
        # 层归一化
        self.layer_norm = nn.LayerNorm(input_dim)
        
        # 前馈网络
        self.feed_forward = nn.Sequential(
            nn.Linear(input_dim, input_dim * 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(input_dim * 4, input_dim)
        )
        
        # DeepAttnMISL
        self.deepattnmisl = DeepAttnMISL(input_dim, num_clusters, size_arg, dropout)

    def forward(self, x1, x2):
        # 交叉注意力融合
        x1_t = x1.transpose(0, 1)  # (n1, batch_size, input_dim)
        x2_t = x2.transpose(0, 1)  # (n2, batch_size, input_dim)
        
        # 使用 x1 作为查询，x2 作为键和值
        attn_output, _ = self.cross_attention(x1_t, x2_t, x2_t)
        # 残差连接和层归一化
        #fused_features = self.layer_norm(x1_t + attn_output)
        # 前馈网络
        fused_features = fused_features + self.feed_forward(fused_features)
        
        # 转置回原始形状
        fused_features = fused_features.transpose(0, 1)  # (batch_size, n1, input_dim)
        # 应用DeepAttnMISL
        output = self.deepattnmisl(fused_features)
        return output

class DAMISLFusion(nn.Module):
    def __init__(self, input_dim=512, num_clusters=10, size_arg="small", dropout=0.25):
        super(DAMISLFusion, self).__init__()
        self.transfusion = TransFusion(norm_layer=RMSNorm, dim=input_dim)
        self.deepattnmisl = DeepAttnMISL(input_dim, num_clusters, size_arg, dropout)

    def forward(self, x1, x2):
        # 使用TransFusion融合x1和x2
        fused = self.transfusion(x1, x2)
        
        # 将融合后的结果输入DeepAttnMISL
        #output = self.deepattnmisl(fused)
        
        return fused

class PPEG(nn.Module):
    def __init__(self, dim=512):
        super(PPEG, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 5, 1, 5//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x, H, W):
        B, _, C = x.shape
        cls_token, feat_token = x[:, 0], x[:, 1:]
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        x = self.proj(cnn_feat.cuda()) + cnn_feat.cuda() + self.proj1(cnn_feat.cuda()) + self.proj2(cnn_feat.cuda())
        x = x.flatten(2).transpose(1, 2)
        x = torch.cat((cls_token.unsqueeze(1).cuda(), x), dim=1)
        return x
        
class TMILFusion(nn.Module):
    def __init__(self, dim=512):
        super(TMILFusion, self).__init__()
        self.pos_layer = PPEG(dim=dim)
        self._fc1 = nn.Sequential(nn.Linear(dim, dim), nn.ReLU())
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.layer1 = TransLayer(dim=dim)
        self.layer2 = TransLayer(dim=dim)
        self.norm = nn.LayerNorm(dim)

        self.x2_projection = nn.Linear(dim, dim)
        self.fusion_layer = TransLayer(dim=dim)

    def forward(self, x1, x2):
        x1, x2 = x1.cuda(), x2.cuda()
        h = self._fc1(x1)
        
        x2_proj = self.x2_projection(x2)
        
        h = torch.cat([h, x2_proj], dim=1)
        h = self.fusion_layer(h)
        
        h = h[:, :x1.shape[1], :]
        
        H = h.shape[1]
        _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
        add_length = _H * _W - H
        h = torch.cat([h, h[:,:add_length,:]], dim=1)

        B = h.shape[0]
        cls_tokens = self.cls_token.expand(B, -1, -1).cuda()
        h = torch.cat((cls_tokens, h), dim=1)

        h = self.layer1(h)
        h = self.pos_layer(h, _H, _W)
        h = self.layer2(h)

        h = self.norm(h)
        
        h = h[:, 1:x1.shape[1]+1, :]

        return h
