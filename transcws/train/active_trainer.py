# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 19:43
# @Author  : Xiaoyu Xing
# @File    : active_trainer.py

# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 21:39
# @Author  : Xiaoyu Xing
# @File    : sequence_labeling_trainer.py

import torch
from progressbar import *
from ..utils import *
import numpy as np
import torch.optim as optim
import torch.nn.functional as F


class ActiveTrainer(object):
    ## TODO
    def __init__(self, **kwargs):
        """
        data_iter_train
        data_iter_dev(None)
        data_iter_exp
        prev_model
        model
        learning_rate
        l2_rate
        momentum
        clip
        path_save_model
        nb_epoch
        max_patience
        label2id_dict
        has_dev
        rules
        topK
        loss_threshold
        :param kwargs:
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def fit(self):
        """
        train model
        :return:
        """

        # get origin loss on train set
        loss1 = []
        self.model.eval()

        for i, feed_dict in enumerate(self.data_iter_train):
            feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
            _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
            labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
            mask_word = labels > 0

            # mask_word = labels > 0
            logits = self.model(_feed_tensor)
            loss = self.model.loss(logits, labels, mask_word)
            loss1.append(loss.data)
        _loss1 = np.mean(np.array(loss1))
        print(('origin loss: {0}').format(
            _loss1))  # loss on all data after fitting the examples

        # fit examples
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                                    weight_decay=self.l2_rate)
        loss_threshold = self.loss_threshold

        for epoch in range(self.nb_epoch):
            train_loss = []
            self.model.train()

            for i, feed_dict in enumerate(self.data_iter_train):
                self.optimizer.zero_grad()
                feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                labeled_mask = self.tensor_from_numpy(feed_dict['exp_words'], use_cuda=self.model.use_cuda)

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
            print(('Epoch: {0}, loss on examples: {0}').format(epoch + 1, _train_loss))  # loss on examples

            if _train_loss < loss_threshold:
                return

        # # loss after fitting examples
        # loss2 = []
        # self.model.eval()
        # for i, feed_dict in enumerate(self.data_iter_train):
        #     self.optimizer.zero_grad()
        #     feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
        #     _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
        #     labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
        #     mask_word = labels > 0
        #
        #     # mask_word = labels > 0
        #     logits = self.model(_feed_tensor)
        #     loss = self.model.loss(logits, labels, mask_word)
        #     loss2.append(loss.data)
        # _loss2 = np.mean(np.array(loss2))
        # print(('loss after fitting examples: {0}').format(
        #     _loss2))  # loss on all data after fitting the examples

        #TODO
        for round in range(10):

            change_masks = []
            history = []
            now = []
            prev_loss = []

            # get per data loss, change topK loss data label
            for i, feed_dict in enumerate(self.data_iter_train):
                self.optimizer.zero_grad()
                feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                mask_word = labels > 0
                logits = self.model(_feed_tensor) # model fit the examples
                logits_prev = self.prev_model(_feed_tensor) # origin model

                feats_mask = mask_word.unsqueeze(-1).expand_as(logits)
                feats_ = logits.masked_select(feats_mask.byte()).contiguous().view(-1, 4)
                feats_prev = logits_prev.masked_select(feats_mask.byte()).contiguous().view(-1, 4)
                previous_gold = labels.clone()
                gold_ = previous_gold.masked_select(mask_word.byte()).contiguous()
                assert feats_.size(0) == gold_.size(0)
                gold_ = gold_ - 1

                loss = F.cross_entropy(feats_, gold_, reduce=False)
                loss_prev = F.cross_entropy(feats_prev, gold_, reduce=False)

                delta_loss = loss - loss_prev
                ind_sorted = np.argsort(delta_loss.data).cuda()
                reverse_idx = np.argsort(ind_sorted).cuda()

                gold_sorted = gold_[ind_sorted]

                feats_topk = feats_[ind_sorted][-self.topK:]
                change_gold_topk = torch.argmax(feats_topk, dim=1, keepdim=False).cuda()
                gold_sorted[-self.topK:] = change_gold_topk

                # change to origin order
                gold_sort_origin = gold_sorted[reverse_idx]
                loss = F.cross_entropy(feats_, gold_sort_origin)  # 下降后的loss
                prev_loss.append(loss.data)

                # change gold label
                ndx = np.where(mask_word)
                gold = previous_gold.clone()
                gold[ndx] = gold_sort_origin

                now_gold_ = (gold + 1) * mask_word.long()

                change_mask = (previous_gold != now_gold_)
                change_masks.append(change_mask)
                history.append(previous_gold)
                now.append(now_gold_)
            _prev_loss = np.mean(np.array(prev_loss))  # loss after moving large loss ones
            print(('round: {0} loss after moving high loss examples: {1}').format(round + 1, _prev_loss))

            # use new label to retrain the network
            for epoch in range(self.nb_epoch):
                bar.start()
                train_loss = []
                self.model.train()

                for i, feed_dict in enumerate(self.data_iter_train):
                    bar.update(i)
                    self.optimizer.zero_grad()
                    feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                    _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                    labels = now[i]
                    labeled_mask = change_masks[i]

                    # mask_word = labels > 0
                    logits = self.model(_feed_tensor)
                    if labeled_mask.sum() == 0:
                        continue
                    loss = self.model.loss(logits, labels, labeled_mask)
                    train_loss.append(loss.data)
                    loss.backward()

                    if self.clip > 0:
                        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)

                    self.optimizer.step()
                bar.finish()
                _train_loss = np.mean(np.array(train_loss))
                print(('train loss on new label: {0}').format(_train_loss))

            after_fit_losses2 = []
            self.model.eval()
            for i, feed_dict in enumerate(self.data_iter_train):
                bar.update(i)
                self.optimizer.zero_grad()
                feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                labels = now[i]
                mask_word = labels > 0

                # mask_word = labels > 0
                logits = self.model(_feed_tensor)
                loss = self.model.loss(logits, labels, mask_word)
                after_fit_losses2.append(loss.data)
            _after_fit_losses2 = np.mean(np.array(after_fit_losses2))
            print(('loss after fitting new examples: {0}').format(
                _after_fit_losses2))  # loss on all data after fitting the examples

            if _after_fit_losses2 > _after_fit_losses:
                # not change label
                print('finished training')
                self.save_model()
                break
            else:
                _after_fit_losses = _after_fit_losses2
                # change label
                for i, feed_dict in enumerate(self.data_iter_train):
                    labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                    labels = now[i]

    def predict(self, data_iter):
        labels_pred = []
        labels_gold = []
        for feed_dict in data_iter:
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

        return labels_pred, labels_gold

    def save_model(self):
        """save_model
        """
        torch.save(self.model.state_dict(), self.path_save_model)

    def change_labels(self, seqs, all_labels, rules, label2id_dict, window_size=10):
        mask = np.zeros(all_labels.shape, dtype=int)
        for i, seq in enumerate(seqs):
            j = 0
            while j < len(seq):
                end = j + window_size
                if end >= len(seq):
                    end = len(seq)
                sub_str = seq[j:end]
                k = len(sub_str)
                while k >= 0:
                    if k == 1:
                        break
                    partial_words = sub_str[0:k]
                    if partial_words in rules:
                        labels = []
                        w_len = len(partial_words)
                        if w_len == 0:
                            continue
                        elif w_len == 1:
                            labels.extend([label2id_dict['S']])
                        else:
                            labels.extend([label2id_dict['B']])
                            for k in range(w_len - 2):
                                labels.extend([label2id_dict['M']])
                            labels.extend([label2id_dict['E']])

                        labels = self.tensor_from_numpy(np.array(labels))
                        all_labels[i][j:j + len(partial_words)] = labels
                        mask[i][j:j + len(partial_words)] = 1
                        j = j + len(partial_words) - 1
                        break
                    k -= 1
                j += 1
        return mask

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
