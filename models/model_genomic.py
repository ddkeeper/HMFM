from collections import OrderedDict
from os.path import join
import pdb

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.model_utils import *



##########################
#### Genomic FC Model ####
##########################
class SNN(nn.Module):
    def __init__(self, omic_input_dim: int, model_size_omic: str='small', n_classes: int=4):
        super(SNN, self).__init__()
        self.n_classes = n_classes
        self.size_dict_omic = {'small': [256, 256, 256, 256], 'big': [1024, 1024, 1024, 256]}
        
        ### Constructing Genomic SNN
        hidden = self.size_dict_omic[model_size_omic]
        fc_omic = [SNN_Block(dim1=omic_input_dim, dim2=hidden[0])]
        for i, _ in enumerate(hidden[1:]):
            fc_omic.append(SNN_Block(dim1=hidden[i], dim2=hidden[i+1], dropout=0.25))
        self.fc_omic = nn.Sequential(*fc_omic)
        
        #survival
        self.classifier = nn.Linear(hidden[-1], n_classes)

        #grade
        #激活函数
        self.act_grad = nn.LogSoftmax(dim=1)
        #分类全连接神经网络
        self.classifier_grade = nn.Linear(hidden[-1], 3)
        init_max_weights(self)


    def forward(self, **kwargs):
        x = kwargs['x_omic']
        features = self.fc_omic(x)

        #Survival
        logits = self.classifier(features).unsqueeze(0)
        Y_hat = torch.topk(logits, 1, dim=1)[1]
        hazards = torch.sigmoid(logits)
        S = torch.cumprod(1 - hazards, dim=1)

        #grade
        hazards_grade = self.classifier_grade(features).unsqueeze(0)
        hazards_grade=self.act_grad(hazards_grade)
        #return hazards, S, Y_hat, None, None
        return hazards, S, Y_hat, None, hazards_grade 
    
    '''
    def relocate(self):
            device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

            if torch.cuda.device_count() > 1:
                device_ids = list(range(torch.cuda.device_count()))
                self.fc_omic = nn.DataParallel(self.fc_omic, device_ids=device_ids).to('cuda:0')
            else:
                self.fc_omic = self.fc_omic.to(device)

            self.classifier = self.classifier.to(device)
    s'''