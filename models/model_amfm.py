import torch
from torch import linalg as LA
import torch.nn.functional as F
import torch.nn as nn

from models.model_utils import *
import admin_torch
import sys

#import deqfusion
from models.mm_model import *

#import DAMISLFusion
from models.model_DeepAttnMISL import *

#import CoAFusion
from models.model_CoAttention import *

#import Gating Network
from models.model_Gating import *

#import mixture of fusion for the input to the final prediction layer 
from models.model_mof import *

class SNNFusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()
        self.snn1 = SNN_Block(dim1=dim, dim2=dim)
        self.snn2 = SNN_Block(dim1=dim, dim2=dim)
        self.norm1 = norm_layer(dim)
        self.norm2 = norm_layer(dim)

    def forward(self, x1, x2):
        return self.snn1(self.norm1(x1)) + self.snn2(self.norm2(x2)).mean(dim=1).unsqueeze(1)

class DropX2Fusion(nn.Module):

    def __init__(self, norm_layer=RMSNorm, dim=512):
        super().__init__()

    def forward(self, x1, x2):
        return x1
        
class MoME(nn.Module):
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256, RoutingNetwork="MLP", expert_idx=0, ablation_expert_id=None):
        super().__init__()
        #self.TransFusion = TransFusion(norm_layer, dim)
        #DAMISLFusion
        self.DAMISLFusion = DAMISLFusion(input_dim=dim)
        #BottleneckTransFusion
        #self.BottleneckTransFusion = BottleneckTransFusion(n_bottlenecks, norm_layer, dim)
        #SNNFusion
        self.SNNFusion = SNNFusion(norm_layer, dim)
        #DropX2Fusion
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        #CoAFusion
        self.CoAFusion = CoAFusion(dim=dim)
        experts = [
            self.CoAFusion,
            self.SNNFusion,
            self.DAMISLFusion,
            self.DropX2Fusion,
        ]

        # 如果指定了要删除的专家ID,从列表中删除
        if ablation_expert_id is not None and 0 <= ablation_expert_id < len(experts):
            del experts[ablation_expert_id]

        # 重新构建字典,保证ID连续
        self.routing_dict = {i: expert for i, expert in enumerate(experts)}

        #computation cost estimate
        self.expert_times = {
            0: 3.96,  # CoAFusion
            1: 4.71,  # SNNFusion
            2: 16.35,  # MILFusion
            3: 0.03,  # ZeroFusion
            #4: 30     # DEQFusion
        }
        #which gating network to use
        if RoutingNetwork == "MLP":
            self.routing_network = MLP_Gate(len(self.routing_dict), dim=dim)
        elif RoutingNetwork == "transformer":
            self.routing_network = Transformer_Gate(len(self.routing_dict), dim=dim)
        elif RoutingNetwork == "CNN":
            self.routing_network = CNN1_Gate(len(self.routing_dict), dim=dim)
            #self.routing_network = CNN2_Gate(len(self.routing_dict), dim=dim)
        else:
            #Single Expert
            self.routing_network = None
            self.expert_idx = expert_idx


    def forward(self, x1, x2, hard=False):
        corresponding_net_id = None
        computation_cost = 0
        if self.routing_network:
            logits, y_soft = self.routing_network(x1, x2, hard)
            if hard:
                #MoME
                corresponding_net_id = torch.argmax(logits, dim=1).item()
                x = self.routing_dict[corresponding_net_id](x1, x2)
                
                #estimate computation cost
                for branch_id, time_cost in self.expert_times.items():
                    computation_cost += y_soft[:, branch_id] * time_cost
                
            else:
                corresponding_net_id = -1
                x = torch.zeros_like(x1)
                for branch_id, branch in self.routing_dict.items():
                    x += branch(x1, x2)
        else:
            corresponding_net_id = self.expert_idx
            x = self.routing_dict[corresponding_net_id](x1, x2)
        #print(f"x.shape: {x.shape}")
        #print(f"computation_cost.shape: {computation_cost.shape}")
        return x, computation_cost, corresponding_net_id
        
class MCMoE(nn.Module): #(Multi-modal Cosine Mixture-of-Experts)
    def __init__(self, n_bottlenecks, norm_layer=RMSNorm, dim=256, max_experts=2, ablation_expert_id=None):
        super().__init__()
        # 初始化专家
        self.DAMISLFusion = DAMISLFusion(input_dim=dim)
        self.SNNFusion = SNNFusion(norm_layer, dim)
        self.DropX2Fusion = DropX2Fusion(norm_layer, dim)
        self.CoAFusion = CoAFusion(dim=dim)
        
        experts = [
            self.CoAFusion,
            self.SNNFusion,
            self.DAMISLFusion,
            self.DropX2Fusion,
        ]

        # 如果指定了要删除的专家ID,从列表中删除
        if ablation_expert_id is not None and 0 <= ablation_expert_id < len(experts):
            del experts[ablation_expert_id]

        # 重新构建字典,保证ID连续
        self.routing_dict = {i: expert for i, expert in enumerate(experts)}
        
        # 添加负载计数器
        self.register_buffer('expert_counts', torch.zeros(len(self.routing_dict)))
        self.register_buffer('total_samples', torch.zeros(1))
        self.accumulation_steps = 32  #累积更新步数

        # 添加专家激活数量分布统计
        self.max_experts = max_experts
        self.register_buffer('expert_k_counts', torch.zeros(max_experts + 1))  # 0到max_experts的分布
        self.register_buffer('total_samples_k', torch.zeros(1))
        # 初始化门控网络
        self.routing_network = MM_CosineGate(
            branch_num=len(self.routing_dict),
            dim=dim,
            max_experts=max_experts
        )
        
        # 用于计算负载均衡损失
        self.num_experts = len(self.routing_dict)
        
    def _update_load_counts(self, logits):
        """更新专家负载计数"""
        # 更新每个专家的选择次数
        self.expert_counts += (logits[0] > 0).float()  # batch_size=1，所以用logits[0]
        self.total_samples += 1

    def _update_k_distribution(self, top_k):
        """更新专家激活数量分布统计"""
        for k in range(self.max_experts + 1):
            self.expert_k_counts[k] += torch.sum(top_k == k)
        self.total_samples_k += len(top_k)        

    def _compute_balance_loss(self):
        """计算累积的负载均衡损失"""
        # 计算累积的负载分布
        load = self.expert_counts / self.expert_counts.sum()
        
        # 计算损失
        load_loss = load.pow(2).sum() * self.num_experts
        
        # 重置计数器
        self.expert_counts.zero_()
        self.total_samples.zero_()
        
        return load_loss

    def get_gating_params(self):
        """获取门控网络的相似度矩阵和门限值"""
        return {
            'sim_matrix': self.routing_network.sim_matrix.data.clone(),
            'activation_gates': self.routing_network.gates.data.clone(),
            'expert_k_counts': self.expert_k_counts.clone(),
            'expert_counts': self.expert_counts.clone()
        }

    def forward(self, x1, x2):
        # 获取门控网络的输出
        logits, top_k = self.routing_network(x1, x2)  # logits: [1, num_experts]
        
        # 初始化输出
        outputs = torch.zeros_like(x1)
        
        # 找出被选中的专家（logits > 0的专家）
        selected_experts = torch.where(logits[0] > 0)[0]  # 因为batch_size=1，所以用logits[0]
        num_selected = selected_experts.size(0)  # 获取选中专家的数量
        
        # 只处理被选中的专家
        for expert_id in selected_experts:
            # 获取专家权重
            expert_weight = logits[0, expert_id].unsqueeze(-1)  # [1]
            # 计算专家输出并加权
            expert_output = self.routing_dict[expert_id.item()](x1, x2)
            outputs += expert_weight * expert_output
        outputs /= num_selected
        
        # 计算辅助损失
        self._update_load_counts(logits)

        # 更新专家激活数量分布
        self._update_k_distribution(top_k)
        balance_loss = 0
        
        return outputs, balance_loss, None

    
class MoMETransformer(nn.Module):
    def __init__(self, n_bottlenecks, omic_sizes=[100, 200, 300, 400, 500, 600], n_classes=4,
                 model_size_wsi: str='small', model_size_omic: str='small', dropout=0.25,
                 gating_network='MLP', expert_idx=0, ablation_expert_id=None,
                 mof_gating_network='MLP', mof_expert_idx=0, mof_ablation_expert_id=None, max_experts = 2, route_mode = True):
        super(MoMETransformer, self).__init__()
        self.omic_sizes = omic_sizes
        self.n_classes = n_classes
        self.size_dict_WSI = {"small": [1024, 512, 512], "big": [1024, 512, 384]}
        #self.size_dict_WSI = {"small": [256, 512, 512], "big": [1024, 512, 384]} #brca condensed-special
        self.size_dict_omic = {'small': [512, 512], 'big': [1024, 1024, 1024, 256]}
        self.route_mode = route_mode
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        sig_networks = []
        for input_dim in omic_sizes:
            fc_omic = [SNN_Block(dim1=input_dim, dim2=hidden[0])]
            for i, _ in enumerate(hidden[1:]):
                fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
            sig_networks.append(nn.Sequential(*fc_omic))
        self.sig_networks = nn.ModuleList(sig_networks)
        
        ### FC Layer over WSI bag
        size = self.size_dict_WSI[model_size_wsi]
        #print(size[0], size[1])
        fc = [nn.Linear(size[0], size[1]), nn.ReLU()]
        fc.append(nn.Dropout(0.25))
        self.wsi_net = nn.Sequential(*fc)
        self.routing_network = gating_network
        if self.routing_network == 'CosMLP':
            ###MCMoEs
            self.MoME_genom1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2],  ablation_expert_id=ablation_expert_id, max_experts=max_experts)
            self.MoME_patho1 = MCMoE(n_bottlenecks=n_bottlenecks, dim=size[2],  ablation_expert_id=ablation_expert_id, max_experts=max_experts)
        else:
            ### MoMEs
            self.MoME_genom1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)
            self.MoME_patho1 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)
            #self.MoME_genom2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)
            #self.MoME_patho2 = MoME(n_bottlenecks=n_bottlenecks, dim=size[2], RoutingNetwork=gating_network, expert_idx=expert_idx, ablation_expert_id=ablation_expert_id)

        ###MoF
        self.MoF = MoF(dim=size[2], RoutingNetwork=mof_gating_network, expert_idx=mof_expert_idx, ablation_expert_id=mof_ablation_expert_id)
        '''
        #old version for fusion feature generation
        ### Classifier
        self.multi_layer1 = TransLayer(dim=size[2])
        self.cls_multimodal = torch.rand((1, size[2])).cuda()
        '''
        ##survival_classifier
        self.classifier = nn.Linear(size[2], n_classes)
        ###grade_classifier
        #激活函数
        self.classifier_grade = nn.Linear(size[2], 3)
        self.act_grad = nn.LogSoftmax(dim=1)

    def get_gating_params(self):
        # 获取门控网络的相似度矩阵和阈值参数
        if self.routing_network == 'CosMLP':
            genom1_gp = self.MoME_genom1.get_gating_params()
            patho1_gp = self.MoME_patho1.get_gating_params()
            return genom1_gp, patho1_gp
        return None, None
    
    def _compute_balance_loss(self):
        """计算所有MCMoE模块的负载均衡损失总和"""
        if self.routing_network == 'CosMLP':
            # 只有在使用MCMoE时才需要计算balance loss
            genom1_loss = self.MoME_genom1._compute_balance_loss()
            patho1_loss = self.MoME_patho1._compute_balance_loss()
            return genom1_loss + patho1_loss
        return 0.0  # 对于其他routing network返回0

    def forward(self, **kwargs):
        x_path = kwargs['x_path']
        x_omic = [kwargs['x_omic%d' % i] for i in range(1,7)]
        #print(x_path.shape)
        h_path_bag = self.wsi_net(x_path) ### path embeddings are fed through a FC layer

        h_omic = [self.sig_networks[idx].forward(sig_feat) for idx, sig_feat in enumerate(x_omic)] ### each omic signature goes through it's own FC layer
        h_omic_bag = torch.stack(h_omic) ### omic embeddings are stacked (to be used in co-attention)
        h_path_bag = h_path_bag.unsqueeze(0)
        #h_path_bag = h_path_bag.unsqueeze(0) #one more time for condensed feature
        h_omic_bag = h_omic_bag.unsqueeze(0)
        #h_path_bag.shape: [1, num_patches, 512], h_omic_bag.shape: [1, 6, 512]
        #print(f"h_path_bag_0: {h_path_bag.shape}")
        #print(f"h_omic_bag_0: {h_omic_bag.shape}")
        #h_path_bag, cost_p1, corresponding_net_id_patho1 = self.MoME_patho1(h_path_bag, h_omic_bag, hard=self.route_mode)
        #h_omic_bag, cost_g1, corresponding_net_id_genom1  = self.MoME_genom1(h_omic_bag, h_path_bag, hard=self.route_mode)
        
        #MCMoE
        if self.routing_network == 'CosMLP':
            h_path_bag, cost_p1, corresponding_net_id_patho1 = self.MoME_patho1(h_path_bag, h_omic_bag)
            h_omic_bag, cost_g1, corresponding_net_id_genom1  = self.MoME_genom1(h_omic_bag, h_path_bag)
        else:
            h_path_bag, cost_p1, corresponding_net_id_patho1 = self.MoME_patho1(h_path_bag, h_omic_bag, hard=True)
            h_omic_bag, cost_g1, corresponding_net_id_genom1 = self.MoME_genom1(h_omic_bag, h_path_bag, hard=True)
            #print(f"h_path_bag_1: {h_path_bag.shape}")
            #print(f"h_omic_bag_1: {h_omic_bag.shape}")
            #h_path_bag, cost_p2 = self.MoME_patho2(h_path_bag, h_omic_bag, hard=True)
            #h_omic_bag, cost_g2 = self.MoME_genom2(h_omic_bag, h_path_bag, hard=True)
        
        #total_time_cost = cost_p1 + cost_g1 + cost_p2 + cost_g2 
        total_time_cost = cost_p1 + cost_g1
        #print(f"h_path_bag_2: {h_path_bag.shape}")
        #print(f"h_omic_bag_2: {h_omic_bag.shape}")

        #MOF: new version for fusion feature generation
        h, jacobian_loss, corresponding_net_id_fuse = self.MoF(h_path_bag, h_omic_bag, hard=self.route_mode)
        #print(f"h.shape: {h.shape}")
        '''
        h_path_bag = h_path_bag.squeeze()
        h_omic_bag = h_omic_bag.squeeze()
        #print(f"h_path_bag_3: {h_path_bag.shape}")
        #升维
        if h_path_bag.dim() == 1:
            h_path_bag = h_path_bag.unsqueeze(0)
        #h_path_bag.shape: [m, 512], h_omic_bag.shape: [6, 512]
        '''
        '''
        #old version for fusion feature generation
        #Attention layer
        h_multi = torch.cat([self.cls_multimodal, h_path_bag, h_omic_bag], dim=0).unsqueeze(0)
        #print(f"h_multi: {h_multi.shape}")
        h = self.multi_layer1(h_multi)[:,0,:]

        jacobian_loss = None
        '''
        #print(f"h: {h.shape}")

        ### Survival Layer
        logits = self.classifier(h)
        Y_hat = torch.topk(logits, 1, dim = 1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)
        
        attention_scores = {}
        expert_choices = {"corresponding_net_id_patho1": corresponding_net_id_patho1,
                          "corresponding_net_id_genom1": corresponding_net_id_genom1,
                          "corresponding_net_id_fuse": corresponding_net_id_fuse}
        ### Grade Layer
        #hazards_grade = self.classifier_grade(h).unsqueeze(0)
        hazards_grade = self.classifier_grade(h)
        hazards_grade = self.act_grad(hazards_grade)
        #print(hazards_grade, hazards_grade.shape)
        #sys.exit()
        
        return hazards, S, Y_hat, attention_scores, hazards_grade, jacobian_loss, total_time_cost, expert_choices
