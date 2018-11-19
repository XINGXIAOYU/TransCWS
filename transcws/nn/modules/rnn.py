# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 15:49
# @Author  : Xiaoyu Xing
# @File    : rnn.py

import torch
import torch.nn as nn


class RNN(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            rnn_unit_type: str, options:['lstm']
            input_dim: rnn input dim
            hidden_unit
            num_layers
            bi_flag: bidirectional flag
        """
        super(RNN, self).__init__()
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        if self.rnn_unit_type == 'lstm':
            self.rnn = nn.LSTM(self.input_dim, self.hidden_unit, self.num_layers, bidirectional=self.bi_flag)

    def forward(self, feats):
        """

        :param feats: [bs, max_len, input_dim]
        :return: [bs, max_len, self.rnn_unit_num]
        """

        rnn_outputs, _ = self.rnn(feats)
        return rnn_outputs
