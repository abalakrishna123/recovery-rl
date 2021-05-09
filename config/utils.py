import numpy as np
import torch
from torch import nn as nn
from scipy.stats import truncnorm

def swish(x):
    return x * torch.sigmoid(x)


def truncated_normal(size, std):
    val = truncnorm.rvs(-2, 2, size=size) * std
    return torch.tensor(val, dtype=torch.float32)


def get_affine_params(ensemble_size, in_features, out_features):

    w = truncated_normal(size=(ensemble_size, in_features, out_features),
                         std=1.0 / (2.0 * np.sqrt(in_features)))
    w = nn.Parameter(w)

    b = nn.Parameter(
        torch.zeros(ensemble_size, 1, out_features, dtype=torch.float32))

    return w, b
