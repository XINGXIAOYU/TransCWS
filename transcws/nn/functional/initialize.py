# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 15:43
# @Author  : Xiaoyu Xing
# @File    : initialize.py

import torch
import numpy as np
import torch.nn as nn


def init_embedding(input_embedding, seed=1337):
    torch.manual_seed(seed)
    scope = np.sqrt(3.0 / input_embedding.size(1))
    nn.init.uniform_(input_embedding, -scope, scope)
