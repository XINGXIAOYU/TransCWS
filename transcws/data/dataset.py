# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:32
# @Author  : Xiaoyu Xing
# @File    : dataset.py

import numpy as np
import random
import math


class DataUtil(object):
    def __init__(self, data_count, data_object, data_names, batch_size, seed=1337, data_type_dict=None):
        """
        split data utils
        :param data_count:
        :param data_object:
        :param data_names:
        :param batch_size:
        :param seed:
        """

        self._data_count = data_count
        self._data_ids = list(range(self._data_count))
        self._data_object = data_object
        self._data_names = data_names
        self._batch_size = batch_size
        self._seed = seed
        self._data_type_dict = data_type_dict

        if not self._data_type_dict:
            self._data_type_dict = dict()
            for data_name in self._data_names:
                self._data_type_dict[data_name] = np.int32

    def split_dataset(self, proportions=(4, 1), shuffle=False):
        """
        split dataset, default proportion is train:dev = 4:1
        :param proportions:
        :param shuffle:
        :return:
        """

        if shuffle:
            random.seed(self._seed)
            random.shuffle(self._data_ids)

        proportions_ = np.array(proportions) / float(sum(proportions))
        data_sizes = (proportions_ * self._data_count).astype(np.int32)
        data_iter_list = []
        current_count = 0

        for i in range(len(proportions)):
            start, end = current_count, current_count + data_sizes[i]

            data_iter = Data_Iter(data_sizes[i], self._data_object, self._data_names,
                                  self._batch_size, self._seed, self._data_type_dict)

            data_iter.data_ids = self._data_ids[start:end]
            data_iter_list.append(data_iter)
            current_count = end

        return data_iter_list


class Data_Iter(object):
    def __init__(self, data_count, data_object, data_names, batch_size, seed=1337, data_type_dict=None):
        """
        Data Iterator
        :param data_count:
        :param data_object:
        :param data_names:
        :param data_type_dict:
        :param batch_size:
        :param seed:
        :param model_type:
        """
        self._data_count = data_count
        self._data_ids = list(range(self._data_count))
        self._data_object = data_object
        self._data_names = data_names
        self._batch_size = batch_size
        self._seed = seed
        self._data_type_dict = data_type_dict
        self._iter_count = math.ceil(self._data_count / float(self._batch_size))

        self._iter_variable = 0

        if not self._data_type_dict:
            self._data_type_dict = dict()
            for data_name in self._data_names:
                self._data_type_dict[data_name] = np.int32

    def shuffle(self):
        random.seed(self._seed)
        random.shuffle(self._data_ids)

    def _generate_batch(self, start, end):
        batch_size = end - start

        batch_max_len_words = max(
            [len(item) for item in self._data_object[self._data_names[0]][self._data_ids[start:end]]])

        try:
            batch_max_len = max(
                [len(item) for item in self._data_object[self._data_names[1]][self._data_ids[start:end]]])
        except:
            pass
        batch_dict = dict()
        for data_name in self._data_names:
            dtype = self._data_type_dict[data_name]

            if data_name == 'words':
                batch_dict[data_name] = np.zeros((batch_size, batch_max_len_words), dtype=dtype)
            elif data_name == 'labels':
                batch_dict[data_name] = np.zeros((batch_size, batch_max_len), dtype=dtype)
            elif data_name == 'exp_masks':
                batch_dict[data_name] = np.zeros((batch_size, batch_max_len), dtype=dtype)
            else:
                raise ValueError("Wrong data name")
            data_object = self._data_object[data_name]
            for i, item in enumerate(data_object[self._data_ids[start:end]]):
                len_item = len(item)
                batch_dict[data_name][i][:len_item] = item

        return batch_dict

    @property
    def iter_count(self):
        return self._iter_count

    @property
    def data_count(self):
        return self._data_count

    @data_count.setter
    def data_count(self, value):
        self._data_count = value

    @property
    def batch_size(self):
        return self._batch_size

    @batch_size.setter
    def batch_size(self, value):
        self._batch_size = value

    @property
    def data_ids(self):
        return self._data_ids

    @data_ids.setter
    def data_ids(self, value):
        self._data_ids = value

    @property
    def iter_variable(self):
        return self._iter_variable

    def __len__(self):
        return self._data_count

    def __iter__(self):
        self._iter_variable = 0
        return self

    def __next__(self):
        start = self._iter_variable
        end = self._iter_variable + self._batch_size
        if end > self._data_count:
            end = self._data_count
        if self._iter_variable > self._data_count or start >= end:
            # self.shuffle()
            raise StopIteration()
        self._iter_variable = end
        return self._generate_batch(start, end)
