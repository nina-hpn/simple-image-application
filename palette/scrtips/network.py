import os
from abc import abstractmethod
import math

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


class GroupNorm32(nn.GroupNorm):
    def forward(self, x):
        return super().forward(x.float()).type(x.dtype)
