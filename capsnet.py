"""
Dynamic Routing Between Capsules
Personal Implementation. Created on 2021/6/30
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
    def __init__(self):
        super().__init__()
        self.input_conv = nn.Sequential(
            nn.Conv2d(1, 256, kernel_size = 9, stride = 1),
            nn.BatchNorm2d(256),
            nn.ReLU(True)
        )
        self.vector_conv = [makeConv(256, 32, 9) for _ in range(8)]
        self.num_primary_caps = 32 * 6 * 6

        # set transformation matrix W
        self.W = nn.parameter(torch.normal(0, 1, (10, self.num_primary_caps, 8, 16)))

    # Conv1 to Primary Caps, making (n, 32, 6, 6, 8) inputs
    def vectorConv(self, x):
        y = self.vector_conv[0](x)
        for i in range(1, 8):
            y = torch.cat([y, self.vector_conv[i](x)], dim = -1)
        return y

    # implementation of dynamic routing algorithm
    def dynamicRouting(self, x):
        pass

    def forward(self, x):
        pass
        
        
