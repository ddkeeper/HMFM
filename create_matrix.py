import csv
from collections import defaultdict

import argparse
import os
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from utils.vis_utils import *  

import argparse
import os
import random
import sys
import os
from timeit import default_timer as timer
import numpy as np

# Internal Imports
from dataset.dataset_survival import Generic_MIL_Survival_Dataset
from utils.file_utils import save_pkl
from utils.core_utils import train
from utils.utils import get_custom_exp_code
import test

# PyTorch Imports
import torch

parser = argparse.ArgumentParser(description='Visualize Expert Similarity Matrix')

### Checkpoint + Misc. Pathing Parameters
parser.add_argument('--data_root_dir',   type=str, default='path/to/data_root_dir',
                    help='Data directory to WSI features (extracted via CLAM')
parser.add_argument('--seed', 			 type=int, default=5,
                    help='Random seed for reproducible experiment (default: 5)')
parser.add_argument('--k', 			     type=int, default=5,
                    help='Number of folds (default: 5)')
parser.add_argument('--k_start',		 type=int, default=-1,
                    help='Start fold (Default: -1, last fold)')
parser.add_argument('--k_end',			 type=int, default=-1,
                    help='End fold (Default: -1, first fold)')
parser.add_argument('--results_dir',     type=str, default='./results',
                    help='Results directory (Default: ./results)')
parser.add_argument('--which_splits',    type=str, default='5foldcv',
                    help='Which splits folder to use in ./splits/ (Default: ./splits/5foldcv')
parser.add_argument('--split_dir',       type=str, default='tcga_blca',
                    help='Which cancer type within ./splits/<which_splits> to use for training. Used synonymously for "task" (Default: tcga_blca)')
parser.add_argument('--log_data',        action='store_true', 
                    help='Log data using tensorboard')
parser.add_argument('--overwrite',     	 action='store_true', default=False,
                    help='Whether or not to overwrite experiments (if already ran)')
parser.add_argument('--load_model',        action='store_true',
                    default=False, help='whether to load model')
parser.add_argument('--path_load_model', type=str,
                    default='/path/to/load', help='path of ckpt for loading')
parser.add_argument('--start_epoch',              type=int,
                    default=0, help='start_epoch.')

### Model Parameters.
parser.add_argument('--model_type',      type=str, choices=['snn', 'amil', 'mcat', 'motcat', 'mome', 'amfm'], 
                    default='motcat', help='Type of model (Default: motcat)')
parser.add_argument('--mode',            type=str, choices=['omic', 'path', 'pathomic', 'cluster', 'coattn'],
                    default='coattn', help='Specifies which modalities to use / collate function in dataloader.')
parser.add_argument('--fusion',          type=str, choices=[
                    'None', 'concat'], default='concat', help='Type of fusion. (Default: concat).')
parser.add_argument('--apply_sig',		 action='store_true', default=False,
                    help='Use genomic features as signature embeddings.')
parser.add_argument('--apply_sigfeats',  action='store_true',
                    default=False, help='Use genomic features as tabular features.')
parser.add_argument('--drop_out',        action='store_true',
                    default=True, help='Enable dropout (p=0.25)')
parser.add_argument('--model_size_wsi',  type=str,
                    default='small', help='Network size of AMIL model')
parser.add_argument('--model_size_omic', type=str,
                    default='small', help='Network size of SNN model')
###deqfusion
parser.add_argument('--no_deq', action='store_true', help='Do not use DEQ for feature fusion')
parser.add_argument('--jacobian_weight', default=20, type=float, help='Jacobian loss weight')
parser.add_argument('--num_layers', default=1, type=int, help='Number of layer steps')
parser.add_argument('--f_thres', default=55, type=int, help='Threshold for equilibrium solver')
parser.add_argument('--b_thres', default=56, type=int, help='Threshold for gradient solver')
parser.add_argument('--stop_mode', default='abs', choices=['abs', 'rel'], help='stop mode for solver')
parser.add_argument('--cosine_scheduler', action='store_true', help='Use cosine scheduler for learning rate decay')
parser.add_argument('--use_default_fuse', action='store_true', help='Use default Attention layer for feature fusion')
parser.add_argument('--no_print', action='store_true', help='Do not print results')
parser.add_argument('--solver', default='anderson', choices=['anderson', 'broyden'], help='Fixed point solver')

### Optimizer Parameters + Survival Loss Function
parser.add_argument('--opt',             type=str,
                    choices=['adam', 'sgd'], default='adam')
parser.add_argument('--batch_size',      type=int, default=1,
                    help='Batch Size (Default: 1, due to varying bag sizes)')
parser.add_argument('--gc',              type=int,
                    default=32, help='Gradient Accumulation Step.')
parser.add_argument('--max_epochs',      type=int, default=20,
                    help='Maximum number of epochs to train (default: 20)')
parser.add_argument('--lr',				 type=float, default=2e-4,
                    help='Learning rate (default: 0.0002)')
parser.add_argument('--bag_loss',        type=str, choices=['svm', 'ce', 'ce_surv', 'nll_surv',
                    'cox_surv'], default='nll_surv', help='slide-level classification loss function (default: nll_surv)')
parser.add_argument('--label_frac',      type=float, default=1.0,
                    help='fraction of training labels (default: 1.0)')
parser.add_argument('--bag_weight',      type=float, default=0.7,
                    help='clam: weight coefficient for bag-level loss (default: 0.7)')
parser.add_argument('--reg', 			 type=float, default=1e-5,
                    help='L2-regularization weight decay (default: 1e-5)')
parser.add_argument('--alpha_surv',      type=float, default=0.0,
                    help='How much to weigh uncensored patients')
parser.add_argument('--reg_type',        type=str, choices=['None', 'omic', 'pathomic'],
                    default='None', help='Which network submodules to apply L1-Regularization (default: None)')
parser.add_argument('--lambda_reg',      type=float, default=1e-4,
                    help='L1-Regularization Strength (Default 1e-4)')
parser.add_argument('--weighted_sample', action='store_true',
                    default=True, help='Enable weighted sampling')
parser.add_argument('--early_stopping',  action='store_true',
                    default=False, help='Enable early stopping')
#computation cost loss term
parser.add_argument("--c_reg", type=float, default=1e-4, help="computation cost reg loss weight")

### task_type
parser.add_argument('--task_type',       type=str, choices=['survival','grade','survival_and_grade'], default='survival', help='Type of task (Default: survival)')
### MOTCat Parameters

#parser.add_argument('--bs_micro',      type=int, default=256,
#                    help='The Size of Micro-batch (Default: 256)') ### new
parser.add_argument('--ot_impl', 			 type=str, default='pot-uot-l2',
                    help='impl of ot (default: pot-uot-l2)') ### new
parser.add_argument('--ot_reg', 			 type=float, default=0.1,
                    help='epsilon of OT (default: 0.1)')
parser.add_argument('--ot_tau', 			 type=float, default=0.5,
                    help='tau of UOT (default: 0.5)')

### MoME Parameters
parser.add_argument('--bs_micro',      type=int, default=4096,
                    help='The Size of Micro-batch (Default: 4096)')
parser.add_argument('--n_bottlenecks', 			 type=int, default=2,
                    help='number of bottleneck features (Default: 2)')
# MoME参数
parser.add_argument('--mome_gating_network', type=str, choices=['MLP', 'transformer', 'CNN', 'CosMLP', 'None'], default='MLP',
                    help='MoME的门控网络结构类型 (默认: MLP)')
parser.add_argument('--mome_expert_idx', type=int, default=0,
                    help='MoME单专家模式下使用的专家ID (默认: 0)')
parser.add_argument('--mome_ablation_expert_id', type=int, default=None,
                    help='MoME专家消融实验中要移除的专家ID (默认: None, 表示不移除任何专家)')

# MoF参数
parser.add_argument('--mof_gating_network', type=str, choices=['MLP', 'transformer', 'CNN', 'CosMLP', 'None'], default='MLP',
                    help='MoF的门控网络结构类型 (默认: MLP)')
parser.add_argument('--mof_expert_idx', type=int, default=0,
                    help='MoF单专家模式下使用的专家ID (默认: 0)')
parser.add_argument('--mof_ablation_expert_id', type=int, default=None,
                    help='MoF专家消融实验中要移除的专家ID (默认: None, 表示不移除任何专家)')
parser.add_argument('--soft_mode', action='store_true', default=True,
                    help='路由模式是软路由还是硬路由 (默认: True, 表示使用软路由)')
#train or test
parser.add_argument('--run_mode', type=str, choices=['train', 'test'], default='train',
                    help='运行模式 (默认: train)')
parser.add_argument('--dataset', type=str, choices=['gbmlgg', 'brca', 'blca', 'luad', 'ucec'], default='gbmlgg',
                    help='数据集选择 (默认: gbmlgg)')

#CosMoME参数
parser.add_argument('--max_experts', type=int, default=2,
                    help='最大激活专家数量 (默认: 2)')

args = parser.parse_args()
args = get_custom_exp_code(args)

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

args.task = '_'.join(args.split_dir.split('_')[:2]) + '_survival'

settings = {'data_root_dir': args.data_root_dir,
            'num_splits': args.k,
            'k_start': args.k_start,
            'k_end': args.k_end,
            'task': args.task,
            'max_epochs': args.max_epochs,
            'results_dir': args.results_dir,
            'lr': args.lr,
            'experiment': args.exp_code,
            'reg': args.reg,
            'label_frac': args.label_frac,
            'bag_loss': args.bag_loss,
            'bag_weight': args.bag_weight,
            'seed': args.seed,
            'model_type': args.model_type,
            'model_size_wsi': args.model_size_wsi,
            'model_size_omic': args.model_size_omic,
            "use_drop_out": args.drop_out,
            'weighted_sample': args.weighted_sample,
            'gc': args.gc,
            'opt': args.opt,
            'bs_micro': args.bs_micro}

print('\nLoad Dataset')
if 'survival' in args.task:
    args.n_classes = 4
    combined_study = '_'.join(args.task.split('_')[:2])
    csv_path= f"./datasets_csv_new/tcga_{args.dataset}_all_clean_three_class.csv"
    dataset = Generic_MIL_Survival_Dataset(#csv_path=csv_path,
                                           #csv_path="/home/yinwendong/AMFM/datasets_csv_new/tcga_gbmlgg_all_clean.csv",
                                           csv_path= csv_path,
                                           mode=args.mode,
                                           apply_sig=args.apply_sig,
                                           #data_dir=args.data_root_dir,
                                           #data_dir= "/home/yinwendong/DEQFusion/experiments/BRCA/features/tcga-brca/mcat_survival_path_features/",
                                           data_dir= f"/home/yinwendong/MCAT-master/data/tcga_{args.dataset}_20x_features/",
                                           shuffle=False,
                                           seed=args.seed,
                                           print_info=True,
                                           patient_strat=False,
                                           n_bins=4,
                                           label_col='survival_months',
                                           ignore=[])
else:
    raise NotImplementedError
# Creates results_dir Directory.
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

# Appends to the results_dir path: 1) which splits were used for training (e.g. - 5foldcv), and then 2) the parameter code and 3) experiment code
exp_code = str(args.exp_code) + '_microb{}'.format(args.bs_micro) + '_s{}'.format(args.seed)

print("===="*30)
print("Experiment Name:", exp_code)
print("===="*30)

#args.results_dir = os.path.join(args.results_dir, args.which_splits, args.param_code, exp_code)
args.results_dir = os.path.join(args.results_dir)
if not os.path.isdir(args.results_dir):
    os.makedirs(args.results_dir)
print("logs saved at ", args.results_dir)

if ('summary_latest.csv' in os.listdir(args.results_dir)) and (not args.overwrite):
    print("Exp Code <%s> already exists! Exiting script." % args.exp_code)
    sys.exit()

# Sets the absolute path of split_dir
args.split_dir = os.path.join('./splits', args.which_splits, args.split_dir)
print("split_dir", args.split_dir)
#assert os.path.isdir(args.split_dir)
settings.update({'split_dir': args.split_dir})

with open(args.results_dir + '/experiment_{}.txt'.format(args.exp_code), 'w') as f:
    print(settings, file=f)
f.close()

print("################# Settings ###################")
for key, val in settings.items():
    print("{}:  {}".format(key, val))

# Create Results Directory
if not os.path.isdir(args.results_dir):
    os.mkdir(args.results_dir)

#1-fold
start_t = timer()
# Gets the Train + Val Dataset Loader.
train_dataset, val_dataset = dataset.return_splits(from_id=False,
                                                    #csv_path='{}/splits_{}.csv'.format(args.split_dir, i)
                                                    csv_path= f"./splits/1foldcv/tcga_{args.dataset}/splits_1.csv")
print('training: {}, validation: {}'.format(
    len(train_dataset), len(val_dataset)))
datasets = (train_dataset, val_dataset)

### Specify the input dimension size if using genomic features.
if 'omic' in args.mode or args.mode == 'cluster' or args.mode == 'graph' or args.mode == 'pyramid':
    args.omic_input_dim = train_dataset.genomic_features.shape[1]
    print("Genomic Dimension", args.omic_input_dim)
elif 'coattn' in args.mode:
    args.omic_sizes = train_dataset.omic_sizes
    print('Genomic Dimensions', args.omic_sizes)
else:
    args.omic_input_dim = 0

def visualize_expert_similarity(model_path, results_dir, args):
    """
    直接从保存的模型文件加载参数并绘制专家相似度矩阵
    
    Args:
        model_path: 模型参数文件路径
        results_dir: 结果保存目录
        args: 模型配置参数
    """
    # 初始化模型
    model_dict = {
        'omic_sizes': args.omic_sizes, 
        'n_classes': args.n_classes, 
        'n_bottlenecks': args.n_bottlenecks,
        'gating_network': args.mome_gating_network,
        'expert_idx': args.mome_expert_idx,
        'ablation_expert_id': args.mome_ablation_expert_id,
        'mof_gating_network': args.mof_gating_network,
        'mof_expert_idx': args.mof_expert_idx,
        'mof_ablation_expert_id': args.mof_ablation_expert_id,
        'max_experts': args.max_experts
    }
    
    from models.model_amfm import MoMETransformer
    model = MoMETransformer(**model_dict)
    
    # 加载模型参数
    model.load_state_dict(torch.load(model_path))
    model = model.cuda()
    
    # 获取门控参数
    genom1_gp, patho1_gp = model.get_gating_params()
    
    # 绘制基因组MoE层的相似度矩阵
    if genom1_gp is not None and 'sim_matrix' in genom1_gp:
        plot_expert_similarity(
            genom1_gp['sim_matrix'], 
            layer_idx='genom1',
            save_path=os.path.join(results_dir, f'genom1_sim_matrix.png')
        )
    
    # 绘制病理图像MoE层的相似度矩阵
    if patho1_gp is not None and 'sim_matrix' in patho1_gp:
        plot_expert_similarity(
            patho1_gp['sim_matrix'],
            layer_idx='patho1',
            save_path=os.path.join(results_dir, f'patho1_sim_matrix.png')
        )

    # 绘制各层的激活阈值
    if genom1_gp is not None and 'activation_gates' in genom1_gp:
        plot_expert_activation(
            genom1_gp['activation_gates'],
            layer_name='genom1',
            save_path=os.path.join(results_dir, f'genom1_activation_gates.png')
        )
    
    if patho1_gp is not None and 'activation_gates' in patho1_gp:
        plot_expert_activation(
            patho1_gp['activation_gates'],
            layer_name='patho1',
            save_path=os.path.join(results_dir, f'patho1_activation_gates.png')
        )
        
# 确保结果目录存在
#os.makedirs(args.results_dir, exist_ok=True)
# 调用可视化函数
visualize_expert_similarity(args.path_load_model, args.results_dir, args)
