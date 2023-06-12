# -*- coding: utf-8 -*-
"""
Created on Thu May 25 18:38:21 2023

@author: nick
"""


from torch import nn
import torch
from torch.nn import functional as F   
 
 
class FocalLoss(nn.Module):
    """ Focal Loss, as described in https://arxiv.org/abs/1708.02002.

    It is an enhancement to cross entropy loss and might be
    useful for classification tasks when there is a large class imbalance.
    x is expected to contain raw, unnormalized scores for each class.
    y is expected to contain class labels.

    Shape:
        - x: (batch_size, C) or (batch_size, C, d1, d2, ..., dK), K > 0.
        - y: (batch_size,) or (batch_size, d1, d2, ..., dK), K > 0.
    Args:
        alpha (Tensor, optional): Weights for each class. Defaults to None.
        gamma (float, optional): A constant, as described in the paper.
            Defaults to 2.
        reduction (str, optional): 'mean', 'sum' or 'none'.
            Defaults to 'mean'.
        ignore_index (int, optional): class label to ignore.
            Defaults to -100.
    """
    def __init__(self, alpha=None, gamma=2, reduction='mean',ignore_index=-100):
        super(FocalLoss, self).__init__()
        self.reduction = reduction
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        self.nll_loss = nn.NLLLoss(
            weight=alpha, reduction='none', ignore_index=ignore_index)
        
    def forward(self, x, y):
        if x.ndim > 2:
            c = x.shape[1]
            x = x.permute(0, *range(2, x.ndim), 1).reshape(-1, c)
            y = y.view(-1)

        unignored_mask = y != self.ignore_index
        y = y[unignored_mask]
        if len(y) == 0:
            return torch.tensor(0.)
        x = x[unignored_mask]

        # compute weighted cross entropy term: -alpha * log(pt)
        # (alpha is already part of self.nll_loss)
        log_p = F.log_softmax(x, dim=-1)
        ce = self.nll_loss(log_p, y)

        # get true class column from each row
        all_rows = torch.arange(len(x))
        log_pt = log_p[all_rows, y]

        # compute focal term: (1 - pt)^gamma
        pt = log_pt.exp()
        focal_term = (1 - pt)**self.gamma

        # the full loss: -alpha * ((1 - pt)^gamma) * log(pt)
        loss = focal_term * ce

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss
 
  
if __name__ == '__main__':
    # unit test
    torch.manual_seed(42)  
    input = torch.randn(7, 7, dtype=torch.float32,
                        requires_grad=True) 
    targets = torch.randint(7, (7, ))
    targets[:] = 4
    print('input: \n', input)
    print('targets: \n', targets)  
 
    criterion = FocalLoss(alpha=torch.FloatTensor([1/2,1/2,1/1,1/1,
                                            1/1,1/2,1/10])) 
    loss = criterion(input, targets)
    loss.backward()  
    cross_en = F.cross_entropy(input, targets)  
    print('focal loss: \n',loss)
    print('cross_entropy: \n',cross_en)