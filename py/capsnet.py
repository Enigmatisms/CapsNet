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
    @staticmethod
    def makeConv(in_chan, out_chan, ksz, stride = 1, pad = 0):
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size = ksz, stride = stride, padding = pad),
            nn.BatchNorm2d(32),
            nn.ReLU(True)
        )

    def __init__(self, num_iter = 5, batch_size = 50):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size = 9, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.vector_conv = [CapsNet.makeConv(256, 32, 9, stride = 2) for _ in range(8)]
        self.num_primary_caps = 32 * 6 * 6
        self.num_iter = num_iter                        # number of dynamic routing iteration
        self.batch_size = batch_size
        # set transformation matrix W
        self.W = nn.Parameter(torch.normal(0, 1, (10, self.num_primary_caps, 8, 16)))

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
        Bs = Var(torch.zeros((self.batch_size, 10, 1, self.num_primary_caps))).cuda()
        x = x.view(self.batch_size, -1, 1, 8)              # dimensionality compression
        u_hat = x[:, None, :, :, :] @ self.W[None, :, :, :, :]
        u_hat = u_hat.squeeze()                             # to (n, 10, 32 * 6 * 6, 16)
        for _ in range(self.num_iter - 1):
            Cs = F.softmax(Bs, dim = 1)
            s = (Cs @ u_hat).squeeze()                           # C is (10, 1, 32 * 6 * 6)
            v = CapsNet.squash(s)                               # (v is n, 10, 16)
            delta_b = (u_hat @ v.unsqueeze(dim = -1)).squeeze()
            Bs = Bs + delta_b.unsqueeze(dim = 2)
        Cs = torch.softmax(Bs, dim = 1)
        s = (Cs @ u_hat).squeeze()                           
        return CapsNet.squash(s)

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
        x = self.vectorConv(x)         # input for primary caps
        return self.dynamicRouting(x)
