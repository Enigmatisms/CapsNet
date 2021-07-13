# -*- coding: utf-8 -*-
"""
    Confusion Matrix Plotting
    @author Sentinel
"""

import torch
import matplotlib.pyplot as plt
import numpy as np

class ConfusionMatrix:
    def __init__(self, class_num):
        self.mat = np.zeros((class_num, class_num))
        self.class_num = class_num

    def addRawElement(self, truth, pred):
        norms = torch.norm(pred.detach(), dim = -1)
        _, idx = norms.max(dim = 1)
        self.mat[truth.cpu().numpy(), idx.cpu().numpy()] += 1

    def addElement(self, truth, pred):
        self.mat[truth, pred] += 1

    def saveConfusionMatrix(self, path):
        plt.cla()
        plt.clf()
        plt.imshow(self.mat, cmap = 'inferno')
        plt.colorbar()
        plt.xlabel("Prediction result")
        plt.ylabel("Ground truth")
        plt.savefig(path)
