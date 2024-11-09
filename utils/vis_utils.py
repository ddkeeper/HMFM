import csv
from collections import defaultdict
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def analyze_expert_choices(samples_expert_choices, output_file='amfm_expert_choices.csv'):
    # 准备 CSV 文件
    fieldnames = ['case_id'] + [f'patho_{i}' for i in range(4)] + [f'genom_{i}' for i in range(4)] + [f'fuse_{i}' for i in range(4)]
    
    with open(output_file, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        # 用于汇总所有样本的计数器
        total_counts = {
            'patho': defaultdict(int),
            'genom': defaultdict(int),
            'fuse': defaultdict(int)
        }

        # 用于计算最后一行汇总数据的字典
        summary_row = {field: 0 for field in fieldnames if field != 'case_id'}

        for case_id, choices in samples_expert_choices.items():
            row = {'case_id': case_id}
            counts = {
                'patho': defaultdict(int),
                'genom': defaultdict(int),
                'fuse': defaultdict(int)
            }

            for submicrobatch in choices:
                counts['patho'][submicrobatch["corresponding_net_id_patho1"]] += 1
                counts['genom'][submicrobatch["corresponding_net_id_genom1"]] += 1
                counts['fuse'][submicrobatch["corresponding_net_id_fuse"]] += 1

            # 填充行数据
            for expert_type in ['patho', 'genom', 'fuse']:
                for i in range(4):  # 假设每种类型最多有4个专家
                    count = counts[expert_type][i]
                    field = f'{expert_type}_{i}'
                    row[field] = count
                    total_counts[expert_type][i] += count
                    summary_row[field] += count  # 累加到汇总行

            writer.writerow(row)

        # 写入汇总行
        summary_row['case_id'] = 'Total'
        writer.writerow(summary_row)

    print(f"Individual sample data and summary row have been written to {output_file}")

    # 打印汇总结果
    print("\nOverall expert selection summary:")
    for expert_type in ['patho', 'genom', 'fuse']:
        print(f"\n{expert_type.capitalize()} experts:")
        total = sum(total_counts[expert_type].values())
        for expert_id, count in total_counts[expert_type].items():
            percentage = (count / total) * 100 if total > 0 else 0
            print(f"  Expert {expert_id}: {count} times ({percentage:.2f}%)")

# 使用方法
#analyze_expert_choices(samples_expert_choices)

def plot_expert_similarity(sim_matrix, layer_idx=0, save_path=None):
    # 计算专家相似度矩阵（在GPU上）
    W = sim_matrix.detach()  # 保持在GPU上
    W_normalized = F.normalize(W, p=2, dim=0)  # 在GPU上进行L2归一化
    similarity = torch.matmul(W_normalized.T, W_normalized)
    
    # 只在最后需要绘图时转到CPU
    similarity = similarity.cpu().numpy()
    
    plt.figure(figsize=(6, 5))
    sns.heatmap(similarity, 
                annot=True,  
                fmt='.3f',   
                cmap='RdBu_r',
                vmin=-0.05,
                vmax=1.0,
                square=True,    
                cbar=False)
    
    plt.title(f'Layer {layer_idx}')
    
    num_experts = similarity.shape[0]
    expert_labels = [f'Expert {i+1}' for i in range(num_experts)]
    plt.xticks(np.arange(num_experts) + 0.5, expert_labels, rotation=45)
    plt.yticks(np.arange(num_experts) + 0.5, expert_labels, rotation=0)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
    plt.show()

def plot_expert_activation(activation_gates, layer_name, save_path):
    """
    可视化单层的专家激活阈值
    
    Args:
        activation_gates: 形状为 [n_experts] 的激活阈值数组
        layer_name: 层的名称，如 'Layer 0'
        save_path: 保存图片的路径
    """
    # 将一维数组重塑为二维数组以用于热力图显示
    # 确保完全转换到CPU并转为numpy数组
    values = activation_gates.detach().cpu().numpy().reshape(-1, 1)
    
    plt.figure(figsize=(3, 6))
    
    # 使用RdBu_r配色方案
    sns.heatmap(values,
                cmap='RdBu_r',
                annot=True,
                fmt='.3f',
                cbar=False,
                xticklabels=[layer_name],
                yticklabels=[f'Expert {i+1}' for i in range(len(activation_gates))],
                square=True)
    plt.title(layer_name)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_expert_k_distribution(k_dist_G1, k_dist_P1, save_path=None):
    """
    绘制专家激活数量分布图
    Args:
        k_dist_G1: G1层的分布计数 (expert_k_counts)
        k_dist_P1: P1层的分布计数 (expert_k_counts)
        save_path: 保存图片的路径
    """
    plt.style.use('seaborn')
    fig, ax = plt.subplots(figsize=(10, 6))

    layers = ['G1', 'P1']
    x = np.arange(len(layers))
    width = 0.15

    def normalize_counts(counts):
        if not isinstance(counts, torch.Tensor):
            counts = torch.tensor(counts)
        total = counts.sum()
        if not isinstance(total, torch.Tensor):
            total = torch.tensor(total)
        return (counts / total).cpu().numpy() if total > 0 else counts.cpu().numpy()

    G1_dist = normalize_counts(k_dist_G1)
    P1_dist = normalize_counts(k_dist_P1)

    # 动态生成颜色列表
    num_experts = max(len(G1_dist), len(P1_dist))
    # 使用 colormap 动态生成颜色
    colors = plt.cm.get_cmap('tab20')(np.linspace(0, 1, num_experts-1))
    
    # 计算实际需要绘制的柱子数量（跳过0）
    active_experts = num_experts - 1
    
    for i in range(1, num_experts):  # 从1开始，跳过0
        expert_G1 = G1_dist[i] if i < len(G1_dist) else 0
        expert_P1 = P1_dist[i] if i < len(P1_dist) else 0
        
        # 调整偏移量计算，使柱子居中对齐
        offset = width * (i - (active_experts + 1)/2)
        plt.bar(x[0] + offset, expert_G1, width, 
                label=f'{i} expert{"s" if i>1 else ""}', 
                color=colors[i-1])
        plt.bar(x[1] + offset, expert_P1, width, 
                color=colors[i-1])

    plt.xlabel('Layers')
    plt.ylabel('Relative Frequency')
    plt.title('Distribution of Number of Activated Experts Per Layer')
    plt.xticks(x, layers)
    plt.legend()

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_expert_distribution(expert_dist_G1, expert_dist_P1, save_path=None):
    """
    绘制专家激活分布饼图
    Args:
        expert_dist_G1: G1层的专家分布 (expert_counts)
        expert_dist_P1: P1层的专家分布 (expert_counts)
        save_path: 保存图片的路径
    """
    plt.style.use('seaborn')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 7))

    # 定义专家名称
    expert_names = ['CoAFusion', 'SNNFusion', 'MILFusion', 'ZeroFusion']
    
    # 根据字典大小动态生成颜色
    num_experts = len(expert_dist_G1)
    colors = plt.cm.Set2(np.linspace(0, 1, num_experts))
    
    def prepare_pie_data(expert_dist):
        if torch.is_tensor(expert_dist):
            expert_dist = {i: v.item() for i, v in enumerate(expert_dist)}
        else:
            expert_dist = {k: v.item() if torch.is_tensor(v) else v 
                          for k, v in expert_dist.items()}
            
        total = sum(expert_dist.values())
        sizes = [count/total for count in expert_dist.values()]
        print(sizes)
        # 过滤掉值为0的部分
        non_zero_sizes = []
        non_zero_colors = []
        for i, size in enumerate(sizes):
            if size > 0:
                non_zero_sizes.append(size)
                non_zero_colors.append(colors[i])
                
        return non_zero_sizes, non_zero_colors

    # 绘制G1层的饼图
    sizes_G1, colors_G1 = prepare_pie_data(expert_dist_G1)
    sizes_G1[0], sizes_G1[1] = sizes_G1[1], sizes_G1[0] #偷袭
    ax1.pie(sizes_G1, 
            autopct='%1.1f%%',
            startangle=90, 
            colors=colors_G1)
    ax1.set_title('Expert Distribution in G1 Layer', 
                  y=-0.03)    # 减小y的绝对值，使标题更靠近饼图

    # 绘制P1层的饼图
    sizes_P1, colors_P1 = prepare_pie_data(expert_dist_P1)
    ax2.pie(sizes_P1,
            autopct='%1.1f%%',
            startangle=90, 
            colors=colors_P1)
    ax2.set_title('Expert Distribution in P1 Layer', 
                  y=-0.03)    # 减小y的绝对值，使标题更靠近饼图

    # 添加图例（显示所有专家，包括未使用的）
    legend_elements = [plt.Rectangle((0,0),1,1, facecolor=colors[i]) 
                      for i in range(num_experts)]
    
    # 将图例放在中间上方，单列显示
    fig.legend(legend_elements, expert_names, 
              loc='center',
              title='Expert Types',
              bbox_to_anchor=(0.5, 0.8))
              
    plt.suptitle('Distribution of Expert Activations', fontsize=14, y=0.95)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()