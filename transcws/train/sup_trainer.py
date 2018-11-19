# -*- coding: utf-8 -*-
# @Time    : 2018/9/4 21:39
# @Author  : Xiaoyu Xing
# @File    : sequence_labeling_trainer.py

import torch
from progressbar import *
from ..utils import *
import numpy as np
import torch.optim as optim



class SupTrainer(object):
    def __init__(self, **kwargs):
        """
        data_iter_train
        data_iter_dev(None)
        model
        learning_rate
        l2_rate
        momentum
        clip
        path_save_model
        nb_epoch
        max_patience
        has_dev

        :param kwargs:
        """
        for k in kwargs:
            self.__setattr__(k, kwargs[k])

    def fit(self):
        """
        train model
        :return:
        """
        best_dev_f1 = -float('inf')
        current_patience = 0
        bar = ProgressBar(maxval=self.data_iter_train.iter_count)

        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.learning_rate,
                                    weight_decay=self.l2_rate)

        for epoch in range(self.nb_epoch):
            bar.start()
            train_loss, dev_loss = [], []
            self.model.train()

            for i, feed_dict in enumerate(self.data_iter_train):
                bar.update(i)
                self.optimizer.zero_grad()
                feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                mask_word = labels > 0
                logits = self.model(_feed_tensor)
                loss = self.model.loss(logits, labels, mask_word)
                train_loss.append(loss.data)
                loss.backward()

                if self.clip > 0:
                    torch.nn.utils.clip_grad_norm(self.model.parameters(), self.clip)

                self.optimizer.step()

            # train f1
            labels_pred, labels_gold = self.predict(self.data_iter_train)
            p, r, f1 = evaluation_01(labels_gold, labels_pred)
            print(("Evaluation on train set: {0}, {1}, {2}").format(p, r, f1))

            # compute dev loss

            if self.has_dev:
                self.model.eval()
                for feed_dict in self.data_iter_dev:
                    feed_tensor = self.tensor_from_numpy(feed_dict['words'], use_cuda=self.model.use_cuda)
                    _feed_tensor = feed_tensor.view(feed_tensor.size(0), -1, 9)
                    labels = self.tensor_from_numpy(feed_dict['labels'], use_cuda=self.model.use_cuda)
                    logits = self.model(_feed_tensor)
                    mask = labels > 0
                    loss = self.model.loss(logits, labels, mask)
                    dev_loss.append(loss.data)

                _train_loss = np.mean(np.array(train_loss))
                _dev_loss = np.mean(np.array(dev_loss))

                print(('Epoch: {0} train loss: {1}, dev loss: {2}').format(epoch + 1, _train_loss, _dev_loss))

                # dev f1
                labels_pred, labels_gold = self.predict(self.data_iter_dev)
                p, r, f1_dev = evaluation_01(labels_gold, labels_pred)
                print(("Evaluation on dev set: {0}, {1}, {2}").format(p, r, f1_dev))

                if f1_dev > best_dev_f1:
                    best_dev_f1 = f1_dev
                    current_patience = 0
                    self.save_model()
                    print(('model has saved to {0}!').format(self.path_save_model))
                else:
                    current_patience += 1
                    print('no improvement, current patience: {0} / {1}'.format(
                        current_patience, self.max_patience))
                    if self.max_patience <= current_patience:
                        print('finished training! (early stopping, max patience: {0})'.format(self.max_patience))
                        return

        bar.finish()
        print('finished training')

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
