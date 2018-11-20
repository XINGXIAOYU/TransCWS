# -*- coding: utf-8 -*-
# @Time    : 2018/11/19 10:31
# @Author  : Xiaoyu Xing
# @File    : finetune.py
import torch
import numpy as np
import torch.nn.functional as F
import codecs
from ..utils import *
import torch.optim as optim


class FineTune(object):
    def __init__(self, **kwargs):
        """
        prev_model
        model
        data_iter_train
        data_iter_dev(None)
        data_iter_exp
        learning_rate
        l2_rate
        momentum
        clip
        path_save_model
        nb_epoch
        max_patience
        loss_threshold
        topK
        path_train
        path_result
        :param kwargs:
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

        self.data_reader = read_data(self.path_train)

    def fit(self):
        # finetune on examples

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                                    weight_decay=self.l2_rate)

        for epoch in range(self.nb_epoch):
            train_loss = []
            self.model.train()
            for i, feed_dict in enumerate(self.data_iter_exp):
                self.optimizer.zero_grad()
                feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                labeled_mask = self.tensor_from_numpy(feed_dict['exp_masks'], use_cuda=self.model.use_cuda)

                logits = self.model(_feed_tensor)
                if labeled_mask.sum() == 0:
                    continue
                loss = self.model.loss(logits, labels, labeled_mask)
                train_loss.append(loss.data)
                loss.backward()

                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                self.optimizer.step()
            _train_loss = np.mean(np.array(train_loss))
            print(('Epoch: {0}, loss on examples: {1}').format(epoch + 1, _train_loss))  # loss on examples

            if _train_loss < self.loss_threshold:
                break

    def get_topK(self, write2file):

        # compare loss
        self.model.eval()
        self.prev_model.eval()
        file_result = codecs.open(self.path_result, 'w', encoding='utf-8')

        delta_losses = [] # save document loss
        seqs = [] # save seqs
        for i, feed_dict in enumerate(self.data_iter_train):
            self.optimizer.zero_grad()
            feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
            _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
            labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
            mask_word = labels > 0

            logits = self.model(_feed_tensor)  # model fit the examples
            logits_prev = self.prev_model(_feed_tensor)  # origin model

            feats_mask = mask_word.unsqueeze(-1).expand_as(logits)
            feats_ = logits.masked_select(feats_mask.byte()).contiguous().view(-1, 2)
            feats_prev = logits_prev.masked_select(feats_mask.byte()).contiguous().view(-1, 2)

            # get label from 1->0
            predict_label = torch.argmax(F.softmax(feats_, dim=1), dim=1)  # this time predict value
            prev_predict_label = torch.argmax(F.softmax(feats_prev, dim=1), dim=1)  # prev time predict value

            ndx = np.where(mask_word)
            prev_predict_label_ = torch.zeros(mask_word.size()).long().cuda()
            prev_predict_label_[ndx] = prev_predict_label

            label_one = prev_predict_label == 1
            label_zero = predict_label == 0
            label_one2zero = label_one & label_zero

            gold_ = labels.masked_select(mask_word.byte()).contiguous()
            assert feats_.size(0) == gold_.size(0)
            gold_ = gold_ - 1

            loss = F.cross_entropy(feats_, gold_, reduce=False)
            loss_prev = F.cross_entropy(feats_prev, gold_, reduce=False)
            delta_loss = loss - loss_prev
            delta_loss = delta_loss * label_one2zero.float()

            ndx = np.where(mask_word)
            delta_loss_ = torch.zeros(mask_word.size()).float().cuda()
            delta_loss_[ndx] = delta_loss


            batch_size = len(logits)
            for i in range(batch_size):
                seq, _ = self.data_reader.__next__()
                length = torch.sum(mask_word[i])
                dl = delta_loss_[i][:length].cpu().data.numpy()
                delta_losses.append(dl)
                seqs.append(seq)

        # whole document
        max_len = max([len(item) for item in seqs])
        delta_losses_ = np.zeros((len(delta_losses), max_len), dtype=float)
        all_mask = np.zeros((len(delta_losses), max_len), dtype=int)

        for i, item in enumerate(delta_losses):
            len_item = len(item)
            delta_losses_[i][:len_item] = item
            all_mask[i][:len_item] = 1

        delta_losses = self.tensor_from_numpy(delta_losses_,'float',use_cuda=self.model.use_cuda)
        all_mask = self.tensor_from_numpy(all_mask,use_cuda=self.model.use_cuda)
        delta_losses_1d = delta_losses.masked_select(all_mask.byte()).contiguous().view(-1) # get loss for all characters

        ind_sorted = np.argsort(delta_losses_1d)
        reverse_idx = np.argsort(ind_sorted)
        mask_loss = torch.zeros(delta_losses_1d.size()).long().cuda()
        mask_loss[-self.topK:] = 1
        mask_loss = mask_loss[reverse_idx]
        ndx = np.where(all_mask)
        topK_mask = torch.zeros(all_mask.size()).long().cuda()
        topK_mask[ndx] = mask_loss
        if write2file:
            # write to file
            for i in range(len(seqs)):
                seq = seqs[i]
                length = torch.sum(all_mask[i])
                m = topK_mask[i][:length].cpu().data.numpy()
                ndx = np.where(m)
                flip_words = []
                if len(ndx[0]) > 0:
                    for dx in ndx[0]:
                        flip_words.append(seq[dx])
                    file_result.write(''.join(seq))
                    file_result.write('\n')
                    file_result.write('****')
                    file_result.write(''.join(flip_words))
                    file_result.write('****')
                    file_result.write('\n')


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
