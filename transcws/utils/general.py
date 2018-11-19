# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:18
# @Author  : Xiaoyu Xing
# @File    : general.py

import numpy as np


def label2id_array(items, voc, oov_id=1):
    """
    project label to id
    :param items:
    :param voc:
    :param oov_id:
    :return:
    """
    arr = np.zeros((len(items),), dtype='int32')
    for i in range(len(items)):
        if items[i] in voc:
            arr[i] = voc[items[i]]
        else:
            arr[i] = oov_id
    return arr


def seqs2id_array(items, voc, oov_id=1, add_binary=True):
    """
    将词序列映射为id序列
    Args:
        items: list, 词序列
        voc: item -> id的映射表
        oov_id: int, 未登录词的编号, default is 1
    Returns:
        arr: np.array, shape=[max_len,]
    """
    if not add_binary:
        arr = np.zeros((len(items),), dtype='int32')
        for i in range(len(items)):
            if items[i] in voc:
                arr[i] = voc[items[i]]
            else:
                arr[i] = oov_id
        return arr

    if add_binary:
        word_idx = []
        _items = [i for i in items]
        _items.append('<EOS>')
        _items.append('<EOS>')
        _items.insert(0, '<BOS>')
        _items.insert(0, '<BOS>')

        for i in range(2, len(_items) - 2):
            for j in range(-2, 3):
                if _items[i + j] in voc:
                    word_idx.append(voc[_items[i + j]])
                else:
                    word_idx.append(oov_id)
            for j in range(-2, 2):

                if _items[i + j] + _items[i + j + 1] in voc:
                    word_idx.append(voc[_items[i + j] + _items[i + j + 1]])
                else:
                    word_idx.append(oov_id)
        arr = np.array(word_idx)
        return arr


def ngram(seq, n=2):
    """
    get bigram words
    :return:
    """
    ngram_list = []
    for i in range(len(seq) - n + 1):
        ngram_list.extend(["".join([m for m in seq[i:i + n]])])
    return ngram_list


def list_generator(lists):
    for i in lists:
        yield i
