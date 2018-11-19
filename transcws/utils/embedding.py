# -*- coding: utf-8 -*-
# @Time    : 2018/11/13 13:10
# @Author  : Xiaoyu Xing
# @File    : embedding.py


import numpy as np


def build_word_embed(word2id_dict, path_embed, seed=1337):
    assert path_embed.endswith('.bin') or path_embed.endswith('.txt')
    word2vec_model, word_dim = load_embed_with_gensim(path_embed)
    word_count = len(word2id_dict)
    np.random.seed(seed)
    scope = np.sqrt(3. / word_dim)
    word_embed_table = np.random.uniform(
        -scope, scope, size=(word_count, word_dim)).astype('float32')

    for word in word2id_dict:
        if len(word) == 1:
            if word in word2vec_model.vocab:
                word_embed_table[word2id_dict[word]] = word2vec_model[word]
    for word in word2id_dict:
        if len(word) == 2:
            word_embed_table[word2id_dict[word]] = (word_embed_table[word2id_dict[word[0]]] + word_embed_table[
                word2id_dict[word[1]]]) / 2

    word_embed_table[0] = np.zeros(word_dim)  # use all zero vectors to be padding vectors
    return word_embed_table


def load_embed_with_gensim(path_embed):
    """
    read pretrained word embeddings
    :param path_embed:
    :return:
    """
    from gensim.models.keyedvectors import KeyedVectors
    if path_embed.endswith('bin'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=True)
    elif path_embed.endswith('txt'):
        word_vectors = KeyedVectors.load_word2vec_format(path_embed, binary=False)
    else:
        raise ValueError('`path_embed` must be `bin` or `txt` file!')
    return word_vectors, word_vectors.vector_size
