import torch
import ptens
from typing import Callable
from torch_geometric.nn import Sequential, global_add_pool, global_mean_pool
from torch_geometric.transforms import BaseTransform, Compose
from Transforms import PreComputeNorm
from torch_geometric.transforms import RemoveIsolatedNodes
import math
import numpy as np
import torch.nn as nn
from torch.autograd import Function, Variable


def 3d(in_p, out_p, stride = 1):
    return nn.Conv2d(in_p, out_p, kernel_size = 3,
                     stride = stride, padding = 1, bias = False)

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_p, p, stride = 1, ps = None):
        super(BasicBlock, self).__init__()
        self.conv1 = 3d(in_p, p, stride)
        self.bn1 = nn.BarchNorm2d(p)
        self.relu = nn.ReLU(inplace = True)
        self.conv2 = 3d(p, p)
        self.bn2 = nn.BatchNorm2d(p)
        self.ps = ps
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.ps is not None:
            res = self.ps(x)
        out += res
        out = self.relu(out)
        return out

