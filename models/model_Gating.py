import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

#MM_CosineGate
import math

from models.model_utils import *
from nystrom_attention import NystromAttention
from unimodals.common_models import GRU, MLP, Transformer, Sequential, Identity
import admin_torch
import sys

def DiffSoftmax(logits, tau=1.0, hard=False, dim=-1):
    y_soft = (logits / tau).softmax(dim)
    if hard:
        # Straight through.
        index = y_soft.max(dim, keepdim=True)[1]
        y_hard = torch.zeros_like(logits, memory_format=torch.legacy_contiguous_format).scatter_(dim, index, 1.0)
        #straight through estimator, y_hard used in feed forward, y_soft used in back propagation
        ret = y_hard - y_soft.detach() + y_soft
    else:
        # Reparametrization trick.
        ret = y_soft
    return ret, y_soft

#old version
class MLP_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(MLP_Gate, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1, x2 = self.fc1(x1), self.fc2(x2)
        x = x1.mean(dim=1) + x2.mean(dim=1)
        #logits = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1) #old version
        logits, y_soft = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=-1)
        return logits, y_soft

#new version MLP+NystromAttention
class SAMLP_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=256):
        super(SAMLP_Gate, self).__init__()
        self.bnum = branch_num
        self.fc1 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        self.fc2 = nn.Sequential(
            *[
                nn.Linear(dim, dim),
                norm_layer(dim),
                nn.GELU(),
            ]
        )
        ### attention mechanism for integrating patho and genom
        self.multi_layer_pg = TransLayer(dim=dim)
        self.cls_pg = torch.rand((1, 1, dim)).cuda()

        self.clsfer = nn.Linear(dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        x1, x2 = self.fc1(x1), self.fc2(x2)
        #print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}")
        #print(f"self.cls_pg.shape: {self.cls_pg.shape}")
        x = torch.cat([self.cls_pg, x1, x2], dim=1)
        #print(f"x.shape: {x.shape}")
        x = self.multi_layer_pg(x)[:,0,:]
        logits, y_soft = DiffSoftmax(self.clsfer(x), tau=temp, hard=hard, dim=1)
        return logits, y_soft

#transformer
class Transformer_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=512):
        super(Transformer_Gate, self).__init__()
        self.bnum = branch_num
        self.transformer = Transformer(dim, 10, 2) # n_features, dim, num_heads=2, num_layers=5
        self.clsfer = nn.Linear(10, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        # x1: batch_size*m1*512, x2: batch_size*m2*512
        x = torch.cat([x1, x2], dim=1)  # batch_size*(m1+m2)*512
        x = self.transformer(x)  # batch_size*(m1+m2)*512
        #x = x.mean(dim=1)  # batch_size*512
        #print(f"x.shape: {x.shape}")
        logits = self.clsfer(x)  # batch_size*branch_num
        logits, y_soft = DiffSoftmax(logits, tau=temp, hard=hard, dim=-1)
        return logits, y_soft

#CNN1：cat+conv+globalpool+fc
class CNN1_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=512, hidden_dim=32):
        super(CNN1_Gate, self).__init__()
        self.bnum = branch_num
        self.conv = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU()
        )
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        # x1: batch_size*m1*dim, x2: batch_size*m2*dim
        #print(f"x1.shape: {x1.shape}, x2.shape: {x2.shape}")
        x = torch.cat([x1, x2], dim=1)  # batch_size*(m1+m2)*dim
        x = x.transpose(1, 2)  # batch_size*dim*(m1+m2)
        #print(f"x.shape: {x.shape}") #1*512*7
        x = self.conv(x)
        x = self.global_pool(x).squeeze(-1)  # batch_size*hidden_dim
        logits = self.fc(x)  # batch_size*branch_num
        logits, y_soft = DiffSoftmax(logits, tau=temp, hard=hard, dim=-1)
        return logits, y_soft

#CNN2：conv+globalpool+fc+average
class CNN2_Gate(nn.Module):
    def __init__(self, branch_num, norm_layer=RMSNorm, dim=512, hidden_dim=32):
        super(CNN2_Gate, self).__init__()
        self.bnum = branch_num
        
        self.conv1 = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            norm_layer(hidden_dim),
            nn.GELU(),
            #nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #norm_layer(hidden_dim),
            #nn.GELU()
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv1d(dim, hidden_dim, kernel_size=3, padding=1),
            norm_layer(hidden_dim),
            nn.GELU(),
            #nn.Conv1d(hidden_dim, hidden_dim, kernel_size=3, padding=1),
            #norm_layer(hidden_dim),
            #nn.GELU()
        )
        
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(hidden_dim, branch_num)

    def forward(self, x1, x2, temp=1.0, hard=False):
        # x1: batch_size*m1*dim, x2: batch_size*m2*dim
        x1 = x1.transpose(1, 2)  # batch_size*dim*m1
        x2 = x2.transpose(1, 2)  # batch_size*dim*m2
        
        x1 = self.conv1(x1)  # batch_size*dim*m1
        x2 = self.conv2(x2)  # batch_size*dim*m2
        
        x1 = self.global_pool(x1).squeeze(-1)  # batch_size*hidden_dim
        x2 = self.global_pool(x2).squeeze(-1)  # batch_size*hidden_dim
        
        x = (x1 + x2)  # batch_size*hidden_dim
        
        logits = self.fc(x)  # batch_size*branch_num
        logits, y_soft = DiffSoftmax(logits, tau=temp, hard=hard, dim=-1)
        return logits, y_soft  # batch_size*branch_num


class SignGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, scores):
        return torch.sign(scores)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class MM_CosineGate(nn.Module):
    def __init__(self, branch_num, dim=256, proj_dim=128, norm_layer=RMSNorm, init_gates=0.5, max_experts=2, init_t=0.07):
        super().__init__()
        self.bnum = branch_num
        self.max_experts = max_experts
        self.clamp_max = torch.log(torch.tensor(1. / 0.01)).item()
        
        self.fc1 = nn.Sequential(
            nn.Linear(dim, proj_dim),
            norm_layer(proj_dim),
            nn.GELU(),
        )
        
        self.fc2 = nn.Sequential(
            nn.Linear(dim, proj_dim),
            norm_layer(proj_dim),
            nn.GELU(),
        )
        
        # 修正相似度矩阵的形状，使其与GAMoEGate一致
        self.register_parameter('sim_matrix', 
            nn.Parameter(torch.nn.init.orthogonal_(
                torch.empty(branch_num, proj_dim)).T.contiguous(), 
            requires_grad=True)
        )
        
        # 每个专家的门限值
        self.gates = nn.Parameter(torch.zeros(branch_num))
        
        # temperature参数
        self.temperature = nn.Parameter(torch.ones(1) * math.log(1 / init_t))
        
        # experts_mask
        self.register_parameter('experts_mask', 
            nn.Parameter(torch.ones(branch_num), requires_grad=False)
        )
        
    def forward(self, x1, x2):
        # MLP处理并归一化
        x1_processed = F.normalize(self.fc1(x1), dim=-1)
        x2_processed = F.normalize(self.fc2(x2), dim=-1)
        
        # 特征融合
        fused_feat = F.normalize((x1_processed.mean(dim=1) + x2_processed.mean(dim=1)) / 2, dim=-1)
        
        # 计算与专家的相似度，添加temperature scaling
        sim_matrix_norm = F.normalize(self.sim_matrix, dim=0)
        logit_scale = torch.clamp(self.temperature, max=self.clamp_max).exp()
        logits = torch.sigmoid(torch.matmul(fused_feat, sim_matrix_norm) * logit_scale)
        
        # 应用experts_mask
        logits = logits * self.experts_mask
        gates = torch.sigmoid(self.gates * logit_scale)
        
        if self.training:
            # 计算差值，用于专家选择
            diff = logits - gates  # [batch_size, num_experts]
            
            logits = F.relu(logits - gates)
            logits = SignGrad.apply(logits)
            top_k = torch.sum(logits > 0, dim=1).to(torch.int)

            # 处理没有选中专家的情况
            zero_mask = (top_k == 0)
            if zero_mask.any():
                # 选择差值最大的专家（最接近阈值的）
                closest_expert = torch.argmax(diff[zero_mask], dim=1)
                logits[zero_mask, closest_expert] = 1.0
                #new_logits[zero_mask, closest_expert] = 1.0
                top_k[zero_mask] = 1
                #logits = new_logits
        else:
            # 计算差值，用于专家选择
            diff = logits - gates  # [batch_size, num_experts]
            
            new_logits = F.relu(diff)
            new_logits = SignGrad.apply(new_logits)
            top_k = torch.sum(new_logits > 0, dim=1).to(torch.int)
            
            # 处理没有选中专家的情况
            zero_mask = (top_k == 0)
            if zero_mask.any():
                # 选择差值最大的专家（最接近阈值的）
                closest_expert = torch.argmax(diff[zero_mask], dim=1)
                new_logits[zero_mask, closest_expert] = 1.0
                top_k[zero_mask] = 1
            
            # 处理选中专家数超过最大值的情况
            over_max_mask = (top_k > self.max_experts)
            if over_max_mask.any():
                for idx in torch.where(over_max_mask)[0]:
                    # 获取当前样本选中的专家索引
                    selected_idx = torch.where(new_logits[idx] > 0)[0]
                    # 根据差值选择top-k个专家
                    expert_diff = diff[idx, selected_idx]
                    _, top_indices = torch.topk(expert_diff, self.max_experts)
                    keep_idx = selected_idx[top_indices]
                    # 重置new_logits
                    new_logits[idx] = 0
                    new_logits[idx, keep_idx] = 1
                    top_k[idx] = self.max_experts
            
            logits = new_logits
        
        #if self.training:
            #print(f'Average Top K is {top_k.float().mean():.2f}, max is {top_k.max()}')
        
        return logits, top_k
