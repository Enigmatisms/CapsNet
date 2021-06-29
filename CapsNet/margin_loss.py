"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Margin loss
"""

import torch
from torch import nn

class MarginLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        pass
    