# -*- coding: utf-8 -*-
# @Time    : 2018/9/11 11:46
# @Author  : Xiaoyu Xing
# @File    : preprocess.py

from __future__ import unicode_literals
import codecs
import re

rNUM = '(-|\+)?\d+((\.|·)\d+)?%?'
rENG = '[A-Za-z_.]+'


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:  # 全角空格直接转换
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):  # 全角字符（除空格）根据关系转化
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring


def preprocess(input_file, output_file, idioms_file=None):
    if idioms_file:
        idioms = dict()
        with codecs.open(idioms_file, 'r', 'utf-8') as f:
            for line in f:
                idioms[line.strip()] = 1

    with codecs.open(input_file, 'r', 'utf-8') as fin:
        with codecs.open(output_file, 'w', 'utf-8') as fout:
            for line in fin:
                sentence = strQ2B(line).strip().split()  # 全角转半角
                new_sentence = []
                for word in sentence:
                    word = re.sub(rNUM, '0', word)  # 将数字替换成0
                    word = re.sub(rENG, 'X', word)  # 将英文替换成X
                    word = re.sub(u'—.*', u'-', word)  # 处理○
                    if idioms_file:
                        if idioms.get(word) is not None:  # 将成语替换成I
                            word = u'I'
                    new_sentence.append(word)
                new_sentence = ' '.join(new_sentence)
                fout.write(new_sentence + '\n')
