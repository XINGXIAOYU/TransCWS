# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 18:48
# @Author  : Xiaoyu Xing
# @File    : active_main.py
import h5py
from optparse import OptionParser
import sys
import yaml
import codecs
from collections import defaultdict
import torch
import os
from transcws.utils import *
from transcws.nn.modules import SupervisedModel


def parse_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str', help='配置文件路径')
    op.add_option(
        '-r', '--rule', dest='rule', type='str', help='规则路径')
    op.add_option('--train', dest='train', action='store_true', default=False, help='训练模式')

    op.add_option('--test', dest='test', action='store_true', default=False, help='测试模式')
    op.add_option(
        '-p', '--preprocess', dest='preprocess', action='store_true', default=False, help='是否进行预处理')

    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()
    if opts.test:
        opts.train = False
    return opts


def load_model(configs):
    """
    加载训练好的模型
    :param configs:
    :return:
    """
    model = init_model(configs)
    path_save_model = configs['data_params']['path_model']

    path_save_model = path_save_model
    model_state = torch.load(path_save_model)
    model.load_state_dict(model_state)
    return model


def init_model(configs):
    # 读取预处理的数据，初始化模型
    root_alphabet = configs['data_params']['alphabet_params']['path']
    feature_size_dict = dict()
    alphabet = read_bin(os.path.join(root_alphabet, 'words.pkl'))
    feature_size_dict['words'] = len(alphabet)

    alphabet = read_bin(os.path.join(root_alphabet, 'labels.pkl'))
    feature_size_dict['labels'] = len(alphabet)

    path_pretrain = configs['data_params']['path_pretrain']
    path_pkl = os.path.join(os.path.dirname(path_pretrain), 'words_embedding.pkl')
    embed = read_bin(path_pkl)
    words_dim = embed.shape[-1]
    pretrained_words_embedding = embed

    # 获取模型参数
    require_grads = configs['model_params']['require_grads']
    rnn_unit_type = configs['model_params']['rnn_type']
    num_rnn_units = configs['model_params']['rnn_units']
    num_layers = configs['model_params']['rnn_layers']
    use_crf = configs['model_params']['use_crf']
    bi_flag = configs['model_params']['bi_flag']

    dropout_rate = configs['model_params']['dropout_rate']
    use_cuda = configs['model_params']['use_cuda']
    average_batch = configs['model_params']['average_batch']

    model = SupervisedModel(feature_size_dict=feature_size_dict, words_dim=words_dim,
                            pretrained_words_embedding=pretrained_words_embedding, require_grad=require_grads,
                            rnn_unit_type=rnn_unit_type, num_rnn_units=num_rnn_units, num_layers=num_layers,
                            bi_flag=bi_flag, dropout_rate=dropout_rate, use_cuda=use_cuda, use_crf=use_crf,
                            average_batch=average_batch, reduce=True)

    if use_cuda:
        model = model.cuda()
    return model
