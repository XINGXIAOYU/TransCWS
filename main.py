# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 12:21
# @Author  : Xiaoyu Xing
# @File    : main.py

from collections import Counter
import os
import h5py
from optparse import OptionParser
import sys
import yaml
import codecs
from collections import defaultdict
import torch
from transcws.utils import *
from transcws.preprocess import *
from transcws.nn.modules import SupervisedModel
from transcws.data import Data_Iter, DataUtil
from transcws.train import SupTrainer, ActiveTrainer, FineTune
from transcws.infer import Predictor


def parse_opts():
    op = OptionParser()
    op.add_option(
        '-c', '--config', dest='config', type='str', help='配置文件路径')
    op.add_option('--train', dest='train', action='store_true', default=False, help='训练模式')
    op.add_option('--active', dest='active', action='store_true', default=False, help='训练active模型')
    op.add_option('--finetune', dest='finetune', action='store_true', default=False, help='finetune模型')
    op.add_option('--test', dest='test', action='store_true', default=False, help='测试模式')
    op.add_option(
        '-p', '--preprocess', dest='preprocess', action='store_true', default=False, help='是否进行预处理')

    op.add_option(
        '--pt', dest='tgtpreprocess', action='store_true', default=False, help='是否对测试集进行预处理')

    op.add_option(
        '-w', '--trainword', dest='trainword', action='store_true', default=False, help='训练集词典')

    argv = [] if not hasattr(sys.modules['__main__'], '__file__') else sys.argv[1:]
    (opts, args) = op.parse_args(argv)
    if not opts.config:
        op.print_help()
        exit()
    if opts.test:
        opts.train = False
    return opts


def extract_info_from_file(path_data, feature_dict, sentence_len=None, add=True):
    """
    读取文件的信息，记录word和label
    :param path_data:
    :param feature_dict:
    :param sentence_len:
    :return:
    """

    data_idx = 0
    for i, (seq, labels) in enumerate(read_data(path_data)):
        if add:
            for word in seq:
                feature_dict['words'].update([word])
            # for label in labels:
            #     feature_dict['labels'].add(label)
            bigram_words = ngram(seq)
            for word in bigram_words:
                feature_dict['biwords'].update([word])
        data_idx += 1
        sentence_len.append(len(seq))
    return data_idx


def data2hdf5(path_data, data_count, token2id_dict, has_label=True):
    """
    save preprocessed data
    :param path_data:
    :param data_count:
    :param token2id_dict:
    :param has_label:
    :return:
    """
    path_hdf5 = path_data + '.hdf5'
    file_hdf5 = h5py.File(path_hdf5, 'w')
    dt = h5py.special_dtype(vlen=np.dtype(np.int32).type)
    dataset_dict = dict()
    dataset = file_hdf5.create_dataset("words", shape=(data_count,), dtype=dt)
    dataset_dict["words"] = dataset
    dataset_label = file_hdf5.create_dataset('labels', shape=(data_count,), dtype=dt)
    dataset_dict['labels'] = dataset_label

    for i, token_list in enumerate(read_data(path_data)):
        tokens = token_list[0]
        token_arr = seqs2id_array(tokens, token2id_dict['words'])
        dataset_dict["words"][i] = token_arr

        if has_label:
            labels = token_list[-1]  # 01
            label_arr = labels
            dataset_dict['labels'][i] = label_arr
        else:
            label_arr = np.zeros(len(token_list[0]))
            dataset_dict['labels'][i] = label_arr

    file_hdf5.close()


def preprocessing(configs):
    """
    文件预处理
    :param configs:
    :return:
    """
    origin_path_train = configs['data_params']['origin_path_train']
    origin_path_dev = configs['data_params']['origin_path_dev'] if 'origin_path_dev' in configs['data_params'] else None
    origin_path_test = configs['data_params']['origin_path_test'] if 'origin_path_test' in configs[
        'data_params'] else None

    path_train = configs['data_params']['path_train']
    path_dev = configs['data_params']['path_dev'] if 'path_dev' in configs['data_params'] else None
    path_test = configs['data_params']['path_test'] if 'path_test' in configs['data_params'] else None

    root_alphabet = configs['data_params']['alphabet_params']['path']
    min_counts = configs['data_params']['alphabet_params']['min_counts']
    path_pretrain = configs['data_params']['path_pretrain']

    PAD = configs['data_params']['pad']
    OOV = configs['data_params']['oov']
    START = configs['data_params']['start']
    END = configs['data_params']['end']

    check_parent_path(path_train)
    preprocess(origin_path_train, path_train)
    if origin_path_dev:
        preprocess(origin_path_dev, path_dev)

    if origin_path_test:
        preprocess(origin_path_test, path_test)

    print('Read files...')
    feature_dict = dict()
    feature_dict['words'] = Counter()
    feature_dict['biwords'] = Counter()
    # feature_dict['labels'] = set()

    sentence_len = list()
    train_data_count = extract_info_from_file(path_train, feature_dict, sentence_len)
    print(("{0}: {1}").format(path_train, train_data_count))

    if path_dev:
        check_parent_path(path_dev)
        dev_data_count = extract_info_from_file(path_dev, feature_dict, sentence_len, add=False)
        print(("{0}: {1}").format(path_dev, dev_data_count))

    if path_test:
        check_parent_path(path_test)
        test_data_count = extract_info_from_file(path_test, feature_dict, sentence_len, add=False)
        print(("{0}: {1}").format(path_test, test_data_count))

    # build label alphabet
    token2id_dict = dict()
    # label2id_dict = dict()
    # for idx, label in enumerate(sorted(feature_dict['labels'])):
    #     label2id_dict[label] = idx + 1
    # label2id_dict['UNK'] = len(label2id_dict) + 1
    # token2id_dict['labels'] = label2id_dict
    # path_label2id_pkl = os.path.join(root_alphabet, 'labels.pkl')
    # check_parent_path(path_label2id_pkl)
    # object2pkl_file(path_label2id_pkl, token2id_dict['labels'])

    # build word alphabet
    word2id_dict = dict()
    word2id_dict[PAD] = 0
    word2id_dict[OOV] = 1
    word2id_dict[START] = 2
    word2id_dict[END] = 3
    start_idx = 4
    for item in sorted(feature_dict['words'].items(), key=lambda d: d[1], reverse=True):
        if item[1] < min_counts[0]:
            continue
        word2id_dict[item[0]] = start_idx
        start_idx += 1
    for item in sorted(feature_dict['biwords'].items(), key=lambda d: d[1], reverse=True):
        if item[1] < min_counts[1]:
            continue
        word2id_dict[item[0]] = start_idx
        start_idx += 1

    token2id_dict['words'] = word2id_dict
    path_word2id_pkl = os.path.join(root_alphabet, 'words.pkl')
    check_parent_path(path_word2id_pkl)
    object2pkl_file(path_word2id_pkl, token2id_dict['words'])

    # build embedding tabel
    print("Getting pretrained word embeddings...")
    if path_pretrain:
        print(("Loading word embedding from {0}").format(path_pretrain))
        word_embed_table = build_word_embed(token2id_dict['words'], path_pretrain)
        path_pkl = os.path.join(os.path.dirname(path_pretrain), 'words_embedding.pkl')
        object2pkl_file(path_pkl, word_embed_table)

    print('Covert data to hdf5')
    data2hdf5(path_train, train_data_count, token2id_dict)

    if path_dev:
        data2hdf5(path_dev, dev_data_count, token2id_dict)
    if path_test:
        data2hdf5(path_test, test_data_count, token2id_dict)


def preprocessing_tgt(configs):
    origin_path_test = configs['data_params']['origin_path_test']
    path_test = configs['data_params']['path_test']

    preprocess(origin_path_test, path_test)

    root_alphabet = configs['data_params']['alphabet_params']['path']
    token2id_dict = dict()
    alphabet = read_bin(os.path.join(root_alphabet, 'words.pkl'))
    token2id_dict['words'] = alphabet

    # alphabet = read_bin(os.path.join(root_alphabet, 'labels.pkl'))
    # token2id_dict['labels'] = alphabet

    data_idx = 0
    for i, (seq, labels) in enumerate(read_data(path_test)):
        data_idx += 1
    test_data_count = data_idx
    print('Covert data to hdf5')
    data2hdf5(path_test, test_data_count, token2id_dict)


def preprocessing_exp(configs):
    examples = configs['rules']['person_path']
    exp_words = configs['rules']['person_name']
    root_alphabet = configs['data_params']['alphabet_params']['path']
    token2id_dict = dict()
    alphabet = read_bin(os.path.join(root_alphabet, 'words.pkl'))
    token2id_dict['words'] = alphabet

    path_hdf5 = examples + '.hdf5'
    check_parent_path(path_hdf5)

    file_hdf5 = h5py.File(path_hdf5, 'w')
    dt = h5py.special_dtype(vlen=np.dtype(np.int32).type)
    dataset_dict = dict()
    dataset = file_hdf5.create_dataset("words", shape=(len(examples),), dtype=dt)
    dataset_dict["words"] = dataset
    dataset_label = file_hdf5.create_dataset('labels', shape=(len(examples),), dtype=dt)
    dataset_dict['labels'] = dataset_label
    dataset_mask = file_hdf5.create_dataset('exp_masks', shape=(len(examples),), dtype=dt)
    dataset_dict['exp_masks'] = dataset_mask

    for i, token_list in enumerate(read_data(examples)):
        tokens = token_list[0]
        seq = "".join(tokens)
        mask = np.zeros(len(tokens), dtype=int)

        j = 0
        while j < len(seq):
            end = j + 5
            if end >= len(seq):
                end = len(seq)
            sub_str = seq[j:end]
            k = len(sub_str)
            while k >= 0:
                if k == 1:
                    break
                partial_words = sub_str[0:k]
                if partial_words in exp_words:
                    mask[j:j + len(partial_words)] = 1
                    j = j + len(partial_words) - 1
                    break
                k -= 1
            j += 1

        token_arr = seqs2id_array(tokens, token2id_dict['words'])
        dataset_dict["words"][i] = token_arr

        labels = token_list[-1]  # 01
        label_arr = labels
        dataset_dict['labels'][i] = label_arr
        dataset_dict['exp_masks'][i] = mask
    file_hdf5.close()


def train_word_dict(configs):
    path_train = configs['data_params']['path_train']
    path_train_word = configs['data_params']['path_train_word']
    train_word_dict = defaultdict(int)
    for i, tokens in enumerate(read_file_content(path_train)):
        for w in tokens:
            train_word_dict[w] += 1

    write_train_words(train_word_dict, path_train_word)


def train_model(configs):
    model = init_model(configs)
    print(model)
    create_dev = configs['model_params']['create_dev']
    batch_size = configs['model_params']['batch_size']

    if create_dev:
        data_iter_train, data_iter_dev = init_train_data(configs,create_dev,batch_size)
        model_trainer = init_trainer(configs, data_iter_train,
                                     data_iter_dev, model)
    else:
        data_iter_train = init_train_data(configs,create_dev,batch_size)
        model_trainer = init_trainer(configs, data_iter_train,
                                     None, model)

    model_trainer.fit()


def train_active_model(configs):
    preprocessing_exp(configs)
    model = load_model(configs)
    prev_model = load_model(configs)
    print(model)
    create_dev = configs['active_model_params']['create_dev']
    batch_size = configs['active_model_params']['batch_size']

    if create_dev:
        data_iter_train, data_iter_dev = init_train_data(configs,create_dev,batch_size)
        data_iter_exp = init_example_data(configs)
        model_trainer = init_active_trainer(configs, data_iter_train,
                                            data_iter_dev, data_iter_exp, model, prev_model)
    else:
        data_iter_train = init_train_data(configs,create_dev,batch_size)
        data_iter_exp = init_example_data(configs)
        model_trainer = init_active_trainer(configs, data_iter_train,
                                            None, data_iter_exp, model, prev_model)

    model_trainer.fit()


def finetune_model(configs):
    preprocessing_exp(configs)
    model = load_model(configs)
    prev_model = load_model(configs)
    print(model)
    create_dev = configs['fine_tune_params']['create_dev']
    batch_size = configs['fine_tune_params']['batch_size']

    if create_dev:
        data_iter_train, data_iter_dev = init_train_data(configs,create_dev,batch_size)
        data_iter_exp = init_example_data(configs)
        model_trainer = init_finetune(configs, data_iter_train,
                                      data_iter_dev, data_iter_exp, model, prev_model)
    else:
        data_iter_train = init_train_data(configs,create_dev,batch_size)
        data_iter_exp = init_example_data(configs)
        model_trainer = init_finetune(configs, data_iter_train,
                                      None, data_iter_exp, model, prev_model)

    model_trainer.fit()
    model_trainer.get_topK(write2file=True)


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
                            average_batch=average_batch)

    if use_cuda:
        model = model.cuda()
    return model


def init_train_data(configs,create_dev,batch_size):
    dev_size = configs['model_params']['dev_size']

    data_names = ['words', 'labels']

    # load train hdf5 file
    path_data = configs['data_params']['path_train'] + '.hdf5'
    train_object_ = h5py.File(path_data, 'r')
    train_object = dict()
    for data_name in data_names:
        train_object[data_name] = train_object_[data_name].value
    train_count = train_object[data_names[0]].size

    if 'path_dev' not in configs['data_params'] or not configs['data_params']['path_dev']:
        if create_dev:
            data_utils = DataUtil(train_count, train_object, data_names, batch_size)
            data_iter_train, data_iter_dev = data_utils.split_dataset(proportions=(1 - dev_size, dev_size),
                                                                      shuffle=True)


        else:
            data_iter_train = Data_Iter(train_count, train_object, data_names, batch_size)
            return data_iter_train

    else:
        path_data = configs['data_params']['path_dev'] + '.hdf5'
        dev_object_ = h5py.File(path_data, 'r')
        dev_object = dict()
        for data_name in data_names:
            dev_object[data_name] = dev_object_[data_name].value
        dev_count = dev_object[data_names[0]].size

        data_iter_dev = Data_Iter(dev_count, dev_object, data_names, batch_size)
        data_iter_train = Data_Iter(train_count, train_object, data_names, batch_size)

    return data_iter_train, data_iter_dev


def init_trainer(configs, data_iter_train, data_iter_dev, model):
    path_save_model = configs['data_params']['path_model']
    check_parent_path(path_save_model)

    nb_epoch = configs['model_params']['nb_epoch']
    max_patience = configs['model_params']['max_patience']

    learning_rate = configs['model_params']['learning_rate']
    l2_rate = configs['model_params']['l2_rate']
    momentum = configs['model_params']['momentum']
    lr_decay = configs['model_params']['lr_decay']
    clip = configs['model_params']['clip']
    # label2id_dict = read_bin(os.path.join(configs['data_params']['alphabet_params']['path'], 'labels.pkl'))

    create_dev = configs['model_params']['create_dev']

    if create_dev:
        trainer = SupTrainer(data_iter_train=data_iter_train, data_iter_dev=data_iter_dev, model=model,
                             learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                             clip=clip,
                             path_save_model=path_save_model, nb_epoch=nb_epoch,
                             max_patience=max_patience, has_dev=True)
    else:
        trainer = SupTrainer(data_iter_train=data_iter_train, model=model,
                             learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                             clip=clip,
                             path_save_model=path_save_model, nb_epoch=nb_epoch,
                             max_patience=max_patience, has_dev=False)
    return trainer


def init_active_trainer(configs, data_iter_train, data_iter_dev, data_iter_exp, model, prev_model):
    path_save_model = configs['data_params']['path_model_active']
    check_parent_path(path_save_model)

    nb_epoch = configs['active_model_params']['nb_epoch']
    max_patience = configs['active_model_params']['max_patience']

    learning_rate = configs['active_model_params']['learning_rate']
    l2_rate = configs['active_model_params']['l2_rate']
    momentum = configs['active_model_params']['momentum']
    lr_decay = configs['active_model_params']['lr_decay']
    clip = configs['active_model_params']['clip']
    # label2id_dict = read_bin(os.path.join(configs['data_params']['alphabet_params']['path'], 'labels.pkl'))

    create_dev = configs['active_model_params']['create_dev']

    rules = configs['rules']['person_name']
    topK = configs['active_model_params']['topK']

    if create_dev:
        trainer = ActiveTrainer(data_iter_train=data_iter_train, data_iter_dev=data_iter_dev,
                                data_iter_exp=data_iter_exp, model=model, prev_model=prev_model,
                                learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                                clip=clip,
                                path_save_model=path_save_model, nb_epoch=nb_epoch,
                                max_patience=max_patience, has_dev=True, rules=rules,
                                topK=topK)
    else:
        trainer = ActiveTrainer(data_iter_train=data_iter_train, data_iter_exp=data_iter_exp, model=model,
                                prev_model=prev_model,
                                learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                                clip=clip,
                                path_save_model=path_save_model, nb_epoch=nb_epoch,
                                max_patience=max_patience, has_dev=False, rules=rules,
                                topK=topK)
    return trainer


def init_finetune(configs, data_iter_train, data_iter_dev, data_iter_exp, model, prev_model):
    path_save_model = configs['data_params']['path_model_active']
    check_parent_path(path_save_model)

    nb_epoch = configs['fine_tune_params']['nb_epoch']
    max_patience = configs['fine_tune_params']['max_patience']

    learning_rate = configs['fine_tune_params']['learning_rate']
    l2_rate = configs['fine_tune_params']['l2_rate']
    momentum = configs['fine_tune_params']['momentum']
    lr_decay = configs['fine_tune_params']['lr_decay']
    clip = configs['fine_tune_params']['clip']

    create_dev = configs['fine_tune_params']['create_dev']

    topK = configs['fine_tune_params']['topK']
    # threshold = configs['fine_tune_params']['threshold']
    loss_threshold = configs['fine_tune_params']['loss_threshold']

    path_result =configs['fine_tune_params']['path_result']
    path_train = configs['data_params']['path_train']


    if create_dev:
        trainer = FineTune(data_iter_train=data_iter_train, data_iter_dev=data_iter_dev,
                           data_iter_exp=data_iter_exp, model=model, prev_model=prev_model,
                           learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                           clip=clip,
                           path_save_model=path_save_model, nb_epoch=nb_epoch,
                           max_patience=max_patience, has_dev=True,
                           topK=topK, loss_threshold=loss_threshold,path_result=path_result,path_train=path_train)
    else:
        trainer = FineTune(data_iter_train=data_iter_train, data_iter_exp=data_iter_exp, model=model,
                           prev_model=prev_model,
                           learning_rate=learning_rate, l2_rate=l2_rate, momentum=momentum, lr_decay=lr_decay,
                           clip=clip,
                           path_save_model=path_save_model, nb_epoch=nb_epoch,
                           max_patience=max_patience, has_dev=False,
                           topK=topK, loss_threshold=loss_threshold,path_result=path_result,path_train=path_train)
    return trainer


def test_model(configs, write2file=True):
    # load model
    model = load_model(configs)

    # init test data
    data_iter_test = init_test_data(configs)

    # init predictor
    path_test = configs['data_params']['path_test']

    if 'path_test_result' not in configs['data_params'] or \
            not configs['data_params']['path_test_result']:
        path_result = configs['data_params']['path_test'] + '.result'

        check_parent_path(path_result)
    else:
        path_result = configs['data_params']['path_test_result']
        check_parent_path(path_result)

    # path_pkl = os.path.join(configs['data_params']['alphabet_params']['path'], 'labels.pkl')
    # label2id_dict = read_bin(path_pkl)

    print(path_test)
    predictor = Predictor(data_iter=data_iter_test, path_result=path_result,
                          path_test=path_test, model=model)

    labels_pred, labels_gold = predictor.predict(write2file)

    p, r, f1 = evaluation_01(labels_gold, labels_pred)

    print(("Evaluation on test set: {0}, {1}, {2}").format(p, r, f1))


def init_test_data(configs):
    """
    初始化测试数据
    :param configs:
    :return:
    """
    batch_size = configs['model_params']['batch_size']
    path_test = configs['data_params']['path_test']
    path_data = path_test + '.hdf5'

    data_names = ['words', 'labels']

    test_object_ = h5py.File(path_data, 'r')
    test_object = dict()
    for data_name in data_names:
        test_object[data_name] = test_object_[data_name].value
    test_count = test_object[data_names[0]].size
    data_iter_test = Data_Iter(test_count, test_object, data_names, batch_size)
    return data_iter_test


def init_example_data(configs):
    path_example = configs['rules']['person_path']
    path_data = path_example + '.hdf5'
    data_names = ['words', 'labels', 'exp_masks']
    exp_object_ = h5py.File(path_data, 'r')
    exp_object = dict()
    for data_name in data_names:
        exp_object[data_name] = exp_object_[data_name].value
    test_count = exp_object[data_names[0]].size
    data_iter_exp = Data_Iter(test_count, exp_object, data_names, test_count)
    return data_iter_exp


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


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "3"

    opts = parse_opts()
    configs = yaml.load(codecs.open(opts.config, encoding='utf-8'))

    if opts.preprocess:
        # 判断是否需要预处理
        preprocessing(configs)

    if opts.tgtpreprocess:
        # 判断是否需要预处理
        preprocessing_tgt(configs)
    if opts.train:  # train
        train_model(configs)
    if opts.active:
        train_active_model(configs)
    if opts.test:  # test
        test_model(configs)
    if opts.trainword:
        train_word_dict(configs)
    if opts.finetune:
        finetune_model(configs)


if __name__ == '__main__':
    main()
