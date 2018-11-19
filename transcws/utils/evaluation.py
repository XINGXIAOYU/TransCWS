# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:12
# @Author  : Xiaoyu Xing
# @File    : evaluation.py

import numpy as np


def evaluation_BMES(labels_gold, labels_pred, label2id_dict, percentage=True):
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels_gold)):
        gold = np.array(labels_gold[i])
        pred = np.array(labels_pred[i])

        pp = pp + np.sum(pred == label2id_dict['E']) + np.sum(pred == label2id_dict['S'])
        np_ = np_ + np.sum(gold == label2id_dict['E']) + np.sum(gold == label2id_dict['S'])

        start = 0
        for j in range(len(gold)):
            if gold[j] == label2id_dict['E'] or gold[j] == label2id_dict['S']:
                flag = True
                for k in range(start, j + 1):
                    if gold[k] != pred[k]:
                        flag = False

                if flag == True:
                    tp += 1
                start = j + 1

    p = float(tp) / float(pp) if tp > 0 and pp > 0 else 0
    r = float(tp) / float(np_) if tp > 0 and pp > 0else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    if percentage:
        p *= 100
        r *= 100
        f1 *= 100
    return p, r, f1



def evaluation_01(labels_gold, labels_pred,percentage=True):
    tp = 0
    np_ = 0
    pp = 0
    for i in range(len(labels_gold)):
        gold = np.array(labels_gold[i])
        pred = np.array(labels_pred[i])

        pp = pp + np.sum(pred == 2)
        np_ = np_ + np.sum(gold == 2)

        start = 0
        for j in range(len(gold)):
            if gold[j] == 2:
                flag = True
                for k in range(start, j + 1):
                    if gold[k] != pred[k]:
                        flag = False

                if flag == True:
                    tp += 1
                start = j + 1

    p = float(tp) / float(pp) if tp > 0 and pp > 0 else 0
    r = float(tp) / float(np_) if tp > 0 and pp > 0else 0
    f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
    if percentage:
        p *= 100
        r *= 100
        f1 *= 100
    return p, r, f1
