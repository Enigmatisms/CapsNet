"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Capsule pathway
"""

import torch
from torch import nn

def makeConv(in_chan, out_chan, ksz, stride = 1, pad = 0):
    return nn.Sequential(
                nn.Conv2d(256, 32, kernel_size = 9, stride = 2),
                nn.BatchNorm2d(32),
                nn.ReLU(True)
            )

class CapsNet(nn.Module):
    def __init__(self, num_iter = 5, batch_size = 50):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size = 9, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.vector_conv = [makeConv(256, 32, 9) for _ in range(8)]
        self.num_primary_caps = 32 * 6 * 6
        self.num_iter = num_iter                        # number of dynamic routing iteration
        self.batch_size = batch_size
        # set transformation matrix W
        self.W = nn.parameter(torch.normal(0, 1, (10, self.num_primary_caps, 8, 16)))

    # Conv1 to Primary Caps, making (n, 32, 6, 6, 8) inputs
    def vectorConv(self, x):
        y = self.vector_conv[0](x)
        for i in range(1, 8):
            y = torch.cat([y, self.vector_conv[i](x)], dim = -1)
        return y

    """
        implementation of dynamic routing algorithm
        the input x is transformed in primary caps
        therefore is (n, 10, 32, 6, 6, 8)
    """
    def dynamicRouting(self, x):
        B = torch.zeros((10, 1, self.num_primary_caps))
        x = x.view(self.batch_size, 10, -1, 1, 8)              # dimensionality compression
        v = torch.zeros(self.batch_size, 10, self.num_primary_caps)
        for _ in range(self.num_iter):
            u_hat = self.W[None, :, :, None, :] @ x
            u_hat = u_hat.squeeze()                             # to (n, 10, 32 * 6 * 6, 16)
            C = torch.softmax(B, dim = 1)
            s = (C @ u_hat).squeeze()
            v = CapsNet.squash(s)                               # (v is n, 10, 32 * 6 * 6)
            delta_b = (u_hat @ v.unsqueeze(dim = -1)).squeeze()
            B += delta_b.unsqueeze(dim = 1)
        return v
    """
        squash: non-linear transformation as activation
        - input has shape (n, 10, num_primary_caps)
    """
    @staticmethod
    def squash(x):
        n = torch.norm(x, dim = -1, keepdim = True)     # shape: (n, 10, 1)
        return n / (n ** 2 + 1.0) * x

    def forward(self, x):
        x = self.input_conv(x)
        x = self.vector_conv(x)         # input for primary caps
        return self.dynamicRouting(x)
