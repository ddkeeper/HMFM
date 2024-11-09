import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
from nystrom_attention import NystromAttention
import admin_torch
import sys

#import deqfusion
from models.mm_model import *

#import DAMISLFusion
from models.model_DeepAttnMISL import *

from models.model_Gating import *

import torch
import torch.nn as nn
#import deqfusion
from models.mm_model import *

class DEQ(nn.Module):
    def __init__(self, dim=512, f_thres=105, b_thres=106, stop_mode='abs', deq=True, num_layers=1, solver='anderson', jacobian_weight=20):
        super(DEQ, self).__init__()
        self.dim = dim
        self.views = 1 + 1
        self.jacobian_weight = jacobian_weight
        self.deq_fusion = DEQFusion(dim, self.views, f_thres, b_thres, stop_mode, deq, num_layers, solver)
        
        # 为path和omic特征分别创建TransLayer
        self.multi_layer_p = TransLayer(dim=dim)
        self.multi_layer_g = TransLayer(dim=dim)
        
        # 创建cls token，现在是三维张量
        self.cls_p = nn.Parameter(torch.randn(1, 1, dim))
        self.cls_g = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_path, x_omic):
        # 处理path特征
        h_multi_p = torch.cat([self.cls_p, x_path], dim=1)
        h_p = self.multi_layer_p(h_multi_p)[:, 0, :]
        
        # 处理omic特征
        h_multi_g = torch.cat([self.cls_g, x_omic], dim=1)
        h_g = self.multi_layer_g(h_multi_g)[:, 0, :]
        
        # 准备DEQ融合的输入
        feature = {0: h_p, 1: h_g}
        fusion_feature = torch.stack([f for f in feature.values()], dim=0).sum(dim=0)
        
        # 执行DEQ融合
        feature_list, jacobian_loss, trace = self.deq_fusion([f for f in feature.values()], fusion_feature)
        
        # 返回融合后的特征
        #print(f"feature_list[-1]: feature_list[-1].shape")
        return feature_list[-1], jacobian_loss # 返回最后一个特征，即融合后的特征以及雅可比损失
    
class SelfAttention(nn.Module):
    def __init__(self, dim=512):
        super(SelfAttention, self).__init__()
        self.dim = dim
        self.multi_layer = TransLayer(dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_path, x_omic):
        # 连接cls_token、path特征和omic特征
        h_multi = torch.cat([self.cls_token, x_path, x_omic], dim=1)
        
        # 应用自注意力
        h = self.multi_layer(h_multi)
        
        # 返回cls_token对应的输出作为融合结果
        return h[:, 0, :], None

class FCN(nn.Module):
    def __init__(self, dim=512):
        super(FCN, self).__init__()
        self.dim = dim
        
        # 为path和omic特征分别创建TransLayer
        self.multi_layer_p = TransLayer(dim=dim)
        self.multi_layer_g = TransLayer(dim=dim)
        
        # 创建cls token
        self.cls_p = torch.rand((1, dim)).cuda()
        self.cls_g = torch.rand((1, dim)).cuda()
        
        # 创建融合层
        self.fusion_layer = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.ReLU(),
            nn.Linear(dim, dim),
            nn.ReLU()
        )
    def forward(self, x_path, x_omic):
        # 处理path特征
        h_multi_p = torch.cat([self.cls_p, x_path], dim=0).unsqueeze(0)
        h_p = self.multi_layer_p(h_multi_p)[:, 0, :]
        
        # 处理omic特征
        h_multi_g = torch.cat([self.cls_g, x_omic], dim=0).unsqueeze(0)
        h_g = self.multi_layer_g(h_multi_g)[:, 0, :]
        
        # 拼接两个模态的全局特征
        h_concat = torch.cat([h_p, h_g], dim=1)
        
        # 通过融合层
        h_fused = self.fusion_layer(h_concat)
        
        return h_fused, None

# 添加仅使用x_path的融合操作
class PathOnly(nn.Module):
    def __init__(self, dim=512):
        super(PathOnly, self).__init__()
        self.dim = dim
        self.multi_layer = TransLayer(dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_path, x_omic):
        # 只使用x_path，忽略x_omic
        h_multi = torch.cat([self.cls_token, x_path], dim=1)
        h = self.multi_layer(h_multi)
        return h[:, 0, :], None

# 添加仅使用x_omic的融合操作
class OmicOnly(nn.Module):
    def __init__(self, dim=512):
        super(OmicOnly, self).__init__()
        self.dim = dim
        self.multi_layer = TransLayer(dim=dim)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

    def forward(self, x_path, x_omic):
        # 只使用x_omic，忽略x_path
        h_multi = torch.cat([self.cls_token, x_omic], dim=1)
        h = self.multi_layer(h_multi)
        return h[:, 0, :], None
        
#mixture of fusion to chooose one fusion operator
class MoF(nn.Module):
    def __init__(self, norm_layer=RMSNorm, dim=512, RoutingNetwork="MLP", expert_idx=0, ablation_expert_id=None):
        super().__init__()
        self.DEQ = DEQ(dim=dim)
        self.SA = SelfAttention(dim=dim)
        self.OmicOnly = OmicOnly(dim=dim)
        self.PathOnly = PathOnly(dim=dim)
        
        experts = [
            self.DEQ,
            self.SA,
            self.OmicOnly,
            self.PathOnly
        ]

        # 如果指定了要删除的专家ID,从列表中删除
        if ablation_expert_id is not None and 0 <= ablation_expert_id < len(experts):
            del experts[ablation_expert_id]

        # 重新构建字典,保证ID连续
        self.routing_dict = {i: expert for i, expert in enumerate(experts)}

        # 选择门控网络
        if RoutingNetwork == "MLP":
            self.routing_network = MLP_Gate(len(self.routing_dict), dim=dim)
        elif RoutingNetwork == "transformer":
            self.routing_network = Transformer_Gate(len(self.routing_dict), dim=dim)
        elif RoutingNetwork == "CNN":
            self.routing_network = CNN1_Gate(len(self.routing_dict), dim=dim)
        else:
            self.routing_network = None  # 单一专家模式
            self.expert_idx = expert_idx

    def forward(self, x_path, x_omic, hard=False):
        #jacobian_loss = None
        if self.routing_network:
            jacobian_loss = None
            logits, y_soft = self.routing_network(x_path, x_omic, hard=hard)
            if hard:
                # 硬选择模式
                corresponding_net_id = torch.argmax(logits, dim=1).item()
                x, jacobian_loss = self.routing_dict[corresponding_net_id](x_path, x_omic)
            else:
                # 软选择模式
                x = torch.zeros(x_path.shape[0], 512).to(x_path.device)  # 初始化输出张量并移至与x_path相同的设备
                corresponding_net_id = -1
                for branch_id, branch in self.routing_dict.items():
                    x1, j_loss = branch(x_path, x_omic)
                    #print(f"x1.shape: {x1.shape}")
                    #print(f"y_soft.shape: {y_soft.shape}")
                    x += x1 * y_soft[:, branch_id]
                    if j_loss:
                        jacobian_loss = j_loss
        else:
            corresponding_net_id = self.expert_idx
            x, jacobian_loss = self.routing_dict[corresponding_net_id](x_path, x_omic)
        return x, jacobian_loss, corresponding_net_id

