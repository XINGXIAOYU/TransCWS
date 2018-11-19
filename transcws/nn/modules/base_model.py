# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 15:54
# @Author  : Xiaoyu Xing
# @File    : supervised_squence_labeling_model.py

import torch
import torch.nn as nn
from .feature import WordFeature
from .rnn import RNN
import torch.nn.functional as F
from .crf import CRF
import numpy as np


class SupervisedModel(nn.Module):
    def __init__(self, **kwargs):
        """
        Args:
            feature_size_dict
            words_dim
            pretrained_words_embedding
            require_grad
            rnn_unit_type
            num_rnn_units
            num_layers
            bi_flag
            dropout_rate
            use_cuda
            use_crf
            average_batch

        :param kwargs:
        """
        super(SupervisedModel, self).__init__()

        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        # word level feature layer

        self.words_feature_layer = WordFeature(words_size=self.feature_size_dict['words'], words_dim=self.words_dim,
                                               pretrained_vectors=self.pretrained_words_embedding,
                                               require_grad=self.require_grad)

        # feature dropout

        self.dropout_feature = nn.Dropout(self.dropout_rate)

        # rnn layer
        rnn_input_dim = self.words_dim * 9
        self.rnn_layer = RNN(rnn_unit_type=self.rnn_unit_type, input_dim=rnn_input_dim, hidden_unit=self.num_rnn_units,
                             num_layers=self.num_layers, bi_flag=self.bi_flag)

        # rnn dropout
        self.dropout_rnn = nn.Dropout(self.dropout_rate)

        # crf layer
        self.target_size = 2
        args_crf = dict({'target_size': self.target_size, 'use_cuda': self.use_cuda})
        args_crf['average_batch'] = self.average_batch
        if self.use_crf:
            self.crf_layer = CRF(**args_crf)

        # dense layer
        hidden_input_dim = (self.num_rnn_units) * 2 if self.bi_flag else self.num_rnn_units

        target_size = self.target_size + 2 if self.use_crf else self.target_size
        self.hidden2tag = nn.Linear(hidden_input_dim, target_size)

        # loss function
        if not self.use_crf:
            self.loss_func = nn.CrossEntropyLoss()
        else:
            self.loss_function = self.crf_layer.neg_log_likelihood_loss

    def forward(self, feed_tensor):
        batch_size = feed_tensor.size(0)
        max_len = feed_tensor.size(1)
        # mask = torch.from_numpy(feed_dict['words'] > 0)
        word_feature = self.words_feature_layer(feed_tensor)
        word_feature = self.dropout_feature(word_feature)
        word_feature = word_feature.view(batch_size, max_len, -1)
        word_feature = torch.transpose(word_feature, 1, 0)

        rnn_outputs = self.rnn_layer(word_feature)
        rnn_outputs = rnn_outputs.transpose(1, 0).contiguous()
        rnn_outputs = self.dropout_rnn(rnn_outputs.view(-1, rnn_outputs.size(-1)))

        rnn_feats = self.hidden2tag(rnn_outputs)
        return rnn_feats.view(batch_size, max_len, -1)

    def loss(self, feats, gold, mask):
        if not self.use_crf:
            feats_mask = mask.unsqueeze(-1).expand_as(feats)
            feats_ = feats.masked_select(feats_mask.byte()).contiguous().view(-1, self.target_size)
            gold_ = gold.masked_select(mask.byte()).contiguous()
            assert feats_.size(0) == gold_.size(0)
            gold_ = gold_ - 1
            loss = self.loss_func(feats_, gold_)
            return loss
        else:
            loss_value = self.loss_function(feats, mask, gold)
            if self.average_batch:
                batch_size = feats.size(0)
                loss_value /= float(batch_size)
            return loss_value

    def predict(self, outputs, actual_len, mask):
        batch_size = outputs.size(0)
        tags_list = []
        _, arg_max = torch.max(F.softmax(outputs, dim=2), dim=2)
        if not self.use_crf:
            arg_max = arg_max + 1  # change 0,1,2,3 to be 1,2,3,4, same as gold label id
            for i in range(batch_size):
                tags_list.append(arg_max[i].cpu().data.numpy()[:actual_len.data[i]])
            return tags_list
        else:
            path_score, best_paths = self.crf_layer(outputs, mask)
            for i in range(batch_size):
                tags_list.append(best_paths[i].cpu().data.numpy()[:actual_len.data[i]])
                # print(best_paths[i].cpu().data.numpy()[:actual_len.data[i]])
            return tags_list
