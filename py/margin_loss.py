"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Margin loss
    @date 2021.6.30
    @author Qianyue He
"""

import torch
from torch import nn

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
        loss = 0.
        x = torch.norm(x, dim = 1)
        loss += torch.max(self.mp - x[y > 0.5], 0) ** 2
        loss += self._lambda * torch.max(x[y <= 0.5] - self.mm, 0) ** 2
        return loss
    