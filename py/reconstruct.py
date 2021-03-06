"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Reconstruction pathway. Somehow confused
    Reconstruction resembles to GAN, which trains two models
    Therefore I suppose the training of reconstruction takes place in a lower frequency
    @date 2021.6.30
    @author Qianyue He
"""

import torch
from torch import nn

class Recons(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Sequential(
            nn.Linear(16, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    """
        input has shape (n, 10, 16)
        before actually run this program, gt label of MNIST is unknown
        I hope it will be (n, 10)
    """
    def forward(self, x, y):
        bsz = x.shape[0]
        masked = x[torch.arange(bsz), y, :]         # output is (n, 16)
        return self.linear(masked)
