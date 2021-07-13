"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Margin loss
    @date 2021.6.30
    @author Qianyue He
"""

import torch
from torch import nn
from torch.functional import norm
import torch.nn.functional as F

class MarginLoss(nn.Module):
    def __init__(self, mp = 0.9, mm = 0.1, _lambda = 0.5):
        super().__init__()
        self.mp = mp
        self.mm = mm
        self._lambda = _lambda

    """
        `x`: shape (n, 10, 16)
        `y`: shape (n, 10) (perhaps)
        This Margin loss is for single digit only
    """
    def forward(self, x, y):
        bsz = x.shape[0]
        masked = torch.zeros(bsz, 10)
        masked[torch.arange(bsz), y] = 1.0
        true_caps = x[masked > 0.5]
        false_caps = x[masked <= 0.5]
        true_norm = F.relu(self.mp - true_caps.norm(), True)
        false_norm = F.relu(false_caps.norm() - self.mm, True)

        loss = 0.
        loss = loss + torch.sum(true_norm ** 2)
        loss = loss + self._lambda * torch.sum(false_norm ** 2)
        return loss

    @staticmethod
    def accCounter(pred:torch.FloatTensor, truth:torch.Tensor):
        norms = torch.norm(pred.detach(), dim = -1)
        _, idx = norms.max(dim = 1)
        rights = torch.sum(idx == truth)
        return rights.item()
