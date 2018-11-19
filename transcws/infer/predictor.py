# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:09
# @Author  : Xiaoyu Xing
# @File    : predictor.py

import torch
import numpy as np
from ..utils import *
import codecs


class Predictor(object):
    def __init__(self, **kwargs):
        """
        model
        data_iter
        path_result
        path_test
        :param kwargs:
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def predict(self, write2file):
        """
        predict
        :return:
        """
        self.model.eval()
        file_result = codecs.open(self.path_result, 'w', encoding='utf-8')
        data_reader = read_data(self.path_test)
        labels_pred = []
        labels_gold = []

        for feed_dict in self.data_iter:
            feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
            _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)

            logits = self.model(_feed_tensor)
            mask, _ = torch.max(_feed_tensor > 0, dim=2)
            actual_len = torch.sum(mask, dim=1).int()
            pred_batch = self.model.predict(logits, actual_len, mask)
            labels_pred.extend(pred_batch)

            gold_batch = np.array(feed_dict['labels']).astype(np.int32)
            _gold_batch = []
            for i in range(len(gold_batch)):
                _gold_batch.append(gold_batch[i][:actual_len.data[i]])
            labels_gold.extend(_gold_batch)

            assert len(_gold_batch) == len(pred_batch)

            if write2file:
                # write to file
                batch_size = len(pred_batch)
                for i in range(batch_size):
                    seq, _ = data_reader.__next__()
                    pred_label = pred_batch[i]

                    seq_result = []

                    start = 0
                    for j in range(len(pred_label)):
                        if pred_label[j] == 2:
                            seq_result.append(''.join(seq[start:j + 1]))
                            start = j + 1

                    line = ' '.join([m for m in seq_result])
                    file_result.write(line)
                    file_result.write('\n')

        return labels_pred, labels_gold

    @staticmethod
    def tensor_from_numpy(data, dtype='long', use_cuda=True):
        """numpy to tensor
        Args:
            data: numpy
            dtype: long or float
            use_cuda: bool
        """
        assert dtype in ('long', 'float')
        if dtype == 'long':
            data = torch.from_numpy(data).long()
        else:
            data = torch.from_numpy(data).float()
        if use_cuda:
            data = data.cuda()
        return data
