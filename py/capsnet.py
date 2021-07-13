"""
    Dynamic Routing Between Capsules
    Personal Implementation. Created on 2021/6/30
    Capsule pathway
    @date 2021.6.30
    @author Qianyue He
"""

import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable as Var


class CapsNet(nn.Module):
    def __init__(self, num_iter = 5, batch_size = 50):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size = 9, stride = 1),
            nn.ReLU(True)
        )
        self.vector_conv = nn.ModuleList([nn.Conv2d(256, 32, 9, stride = 2) for _ in range(8)])
        self.num_primary_caps = 32 * 6 * 6
        self.num_iter = num_iter                        # number of dynamic routing iteration
        self.batch_size = batch_size
        # set transformation matrix W
        self.W = nn.Parameter(torch.normal(0, 1, (10, self.num_primary_caps, 8, 16)))
        self.linear = nn.Sequential(
            nn.Linear(160, 512),
            nn.ReLU(True),
            nn.Linear(512, 1024),
            nn.ReLU(True),
            nn.Linear(1024, 784),
            nn.Sigmoid()
        )

    def setCuda(self):
        for i in range(len(self.vector_conv)):
            self.vector_conv[i] = self.vector_conv[i].cuda()

    # Conv1 to Primary Caps, making (n, 32, 6, 6, 8) inputs
    def vectorConv(self, x):
        y = self.vector_conv[0](x).unsqueeze(dim = -1)
        for i in range(1, 8):
            y = torch.cat([y, self.vector_conv[i](x).unsqueeze(dim = -1)], dim = -1)
        return y

    """
        implementation of dynamic routing algorithm
        the input x is transformed in primary caps
        therefore is (n, 10, 32, 6, 6, 8)
    """
    def dynamicRouting(self, x):
        Bs = Var(torch.zeros((self.batch_size, 10, self.num_primary_caps, 1))).cuda()
        x = x.view(self.batch_size, -1, 1, 8)              # dimensionality compression
        u_hat = x[:, None, :, :, :] @ self.W[None, :, :, :, :]
        u_hat = u_hat.squeeze()                             # to (n, 10, 32 * 6 * 6, 16)
        for _ in range(self.num_iter - 1):
            Cs = F.softmax(Bs, dim = 1)
            s = (Cs * u_hat).sum(dim = 2)                       # C is (n, 10, 1, 32 * 6 * 6)
            v = CapsNet.squash(s)                               # (v is n, 10, 16)
            delta_b = (v[:, :, None, :] * u_hat).sum(dim = -1)
            Bs = Bs + delta_b.unsqueeze(dim = -1)
        Cs = torch.softmax(Bs, dim = 1)
        s = (Cs * u_hat).sum(dim = 2)                          
        return CapsNet.squash(s)

    """
        squash: non-linear transformation as activation
        - input has shape (n, 10, num_primary_caps)
    """
    @staticmethod
    def squash(x):
        n = torch.sum(x ** 2, dim = -1, keepdim = True)
        sqrt_n = torch.sqrt(n)
        return n / (n + 1.0) * x / sqrt_n

    def reconstruct(self, x, y):
        bsz = x.shape[0]
        mask = Var(torch.zeros(bsz, 10)).cuda()
        mask[torch.arange(bsz), y] = 1.0         # output is (n, 16)
        masked = x * mask[:, :, None]
        return self.linear(masked.view(bsz, -1))

    def forward(self, x, y):
        x = self.input_conv(x)
        x = self.vectorConv(x)         # input for primary caps
        x = CapsNet.squash(x)
        x = self.dynamicRouting(x)
        return x, self.reconstruct(x, y)
