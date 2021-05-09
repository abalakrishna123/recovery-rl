'''
Built on on SAC implementation from 
https://github.com/pranz24/pytorch-soft-actor-critic
'''

import os
import cv2
import numpy as np
import plotly
from plotly.graph_objs import Scatter
from plotly.graph_objs.scatter import Line
import math
import torch
from dotmap import DotMap
from config import create_config
''' 
    SAC utils
'''


def create_log_gaussian(mean, log_std, t):
    quadratic = -((0.5 * (t - mean) / (log_std.exp())).pow(2))
    l = mean.shape
    log_z = log_std
    z = l[-1] * math.log(2 * math.pi)
    log_p = quadratic.sum(dim=-1) - log_z.sum(dim=-1) - 0.5 * z
    return log_p


def logsumexp(inputs, dim=None, keepdim=False):
    if dim is None:
        inputs = inputs.view(-1)
        dim = 0
    s, _ = torch.max(inputs, dim=dim, keepdim=True)
    outputs = s + (inputs - s).exp().sum(dim=dim, keepdim=True).log()
    if not keepdim:
        outputs = outputs.squeeze(dim)
    return outputs


'''
    Update target networks for SAC
'''


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) +
                                param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


'''
    Linear schedule for RSPO
'''


def linear_schedule(startval, endval, endtime):
    return lambda t: startval + t / endtime * (endval - startval
                                               ) if t < endtime else endval


'''
    Utility to request a required argument in a dotmap
'''


def get_required_argument(dotmap, key, message, default=None):
    val = dotmap.get(key, default)
    if val is default:
        raise ValueError(message)
    return val


'''
    Utility to construct config for model-based recovery policy
'''


def recovery_config_setup(exp_cfg, logdir):
    ctrl_args = DotMap(**{key: val for (key, val) in exp_cfg.ctrl_arg})
    cfg = create_config(exp_cfg.env_name, "MPC", ctrl_args, exp_cfg.override, logdir)
    cfg.pprint()
    return cfg
