# -*- coding: utf-8 -*-
"""
Created on Tue May  9 10:59:23 2023
(1) 对异常样本敏感: 假如训练集中有个样本label标错了，
那么focal loss会一直放大这个样本的loss(模型想矫正回来），导致网络往错误方向学习。
(2)对分类边界异常点处理不理想：由于边界样本表示相似性较高，
对于不同异常值表示，每次损失更新时，都会有反复在分类决策面（0.5）上反复横跳的点，
导致模型收敛速度下降，退化成交叉熵损失。
@author: nick
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
import torch
from torch.nn import functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha, gamma=2, reduction='mean'):
        """
        input:
            pred: shape -> [bs, num_classes]，without softmax
            target: shape -> [bs], without one_hot encoding
            alpha: FloatTensor of weight for each class
            gamma: control imbalance in difficulties
            reduction: method of loss calculation, can be either mean or sum
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, pred, target):
        alpha = self.alpha[target]  # 为当前batch内的样本，逐个分配类别权重，shape=(bs), 一维向量
        log_softmax = torch.log_softmax(pred, dim=1) # 对模型裸输出做softmax再取log, shape=(bs, 3)
        logpt = torch.gather(log_softmax, dim=1, index=target.view(-1, 1))  # 取出每个样本在类别标签位置的log_softmax值, shape=(bs, 1)
        logpt = logpt.view(-1)  # 降维，shape=(bs)
        ce_loss = -logpt  # 对log_softmax再取负，就是交叉熵了
        pt = torch.exp(logpt)  #对log_softmax取exp，把log消了，就是每个样本在类别标签位置的softmax值了，shape=(bs)
        focal_loss = alpha * (1 - pt) ** self.gamma * ce_loss  # 根据公式计算focal loss，得到每个样本的loss值，shape=(bs)
        if self.reduction == "mean":
            return torch.mean(focal_loss)
        if self.reduction == "sum":
            return torch.sum(focal_loss)
        return focal_loss
    
    
if __name__ == '__main__':
    torch.manual_seed(42) 
    input = torch.randn(5, 6, dtype=torch.float32, requires_grad=True)
    print('input值为\n', input.shape)
    targets = torch.randint(5, (5, ))
    print('targets值为\n', targets.shape)
    
    criterion = FocalLoss()
    loss = criterion(input, targets)
    loss.backward()