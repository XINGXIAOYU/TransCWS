# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 15:29
# @Author  : Xiaoyu Xing
# @File    : feature.py
import torch
import torch.nn as nn
from ..functional import init_embedding
import torch.nn.functional as F

class WordFeature(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            words_size
            words_dim
            pretrained_vectors
            require_grad
        get word embedding
        :param kwargs:
        """
        super(WordFeature, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # feature embedding layer
        embed = nn.Embedding(self.words_size, self.words_dim)
        if self.pretrained_vectors is not None:
            # load pretrained vectors
            embed.weight.data.copy_(torch.from_numpy(self.pretrained_vectors))
        else:
            # initialize randomly
            init_embedding(embed.weight)

        embed.weight.requires_grad = self.require_grad
        self.feature_embedding = embed

    def forward(self, x):
        """

        :param input_dict:
        :return: [bs, max_len, input_size]
        """

        embed_outputs = self.feature_embedding(x)
        return embed_outputs
