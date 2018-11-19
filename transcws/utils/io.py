# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:10
# @Author  : Xiaoyu Xing
# @File    : io.py
import os
import pickle


def read_data(path):
    """
    read file， label words in 01 format, 1 means to split after the word
    :param path: file path
    :return:
    """

    with open(path, "r") as fw:
        for line in fw:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')

            seq = []
            labels = []

            for word in tokens:
                w_len = len(word)
                if w_len == 0:
                    continue
                elif w_len == 1:
                    labels.extend([2])
                else:
                    labels.extend([1])
                    for j in range(w_len - 2):
                        labels.extend([1])
                    labels.extend([2])
                seq.extend([c for c in word])
            yield seq, labels


def check_parent_path(path):
    """
    if parent path not exist then create one
    :param path:
    :return:
    """
    parent_path = os.path.dirname(path)
    if not os.path.exists(parent_path):
        os.makedirs(parent_path)



def object2pkl_file(path_pkl, obj):
    """
    dump object to pickle file
    :param path_pkl:
    :param obj:
    :return:
    """
    with open(path_pkl, "wb") as fw:
        pickle.dump(obj, fw)


def read_bin(path):
    """
    read binary file
    :param path:
    :return:
    """

    f = open(path, 'rb')
    return pickle.load(f)


def read_file_content(path):
    """
    read data split by ' '
    :param path:
    :return:
    """
    with open(path, 'r') as fw:
        for line in fw:
            line = line.strip()
            if len(line) == 0:
                continue
            tokens = line.split(' ')
            yield tokens


def write_seg_result(path, seg_list):
    """
    write result after using forward & backward branching entropy or dictionary
    :param path:
    :param input_str:
    :param branch_pos:
    :return:
    """
    with open(path, 'w') as fw:
        for seg in seg_list:
            res = ' '.join([k for k in seg])
            fw.write(res + '\n')


def write_train_words(train_dict, path):
    with open(path, 'w') as fw:
        for word in train_dict:
            fw.write(word + '\n')
