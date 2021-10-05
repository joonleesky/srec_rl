from ..common.logger import AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim

from abc import *
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os


class AbstractTrainer(metaclass=ABCMeta):
    def __init__(self, args, model, train_loader, val_loader, test_loader, local_exp_path):
        self.args = args
        self.device = args.device
        self.model = model.to(self.device)
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        self.clip_grad_norm = args.clip_grad_norm
        self.criterion = self._create_criterion()

        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        self.local_exp_path = local_exp_path

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay, momentum=args.momentum)
        else:
            raise ValueError
            
    def _create_scheduler(self):
        args = self.args
        return optim.lr_scheduler.StepLR(self.optimizer, step_size=args.decay_step, gamma=args.gamma)
            
    @abstractmethod
    def _create_criterion(self):
        pass
            
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @abstractmethod
    def calculate_loss(self, batch):
        pass

    @abstractmethod
    def calculate_metrics(self, batch):
        pass

    def train(self):
        best_metric_value = 0
        best_epoch = -1
        accum_iters = 0
        stop_training = False
        for epoch in range(self.num_epochs):
            accum_iters = self.train_one_epoch(epoch, accum_iters)
            val_log_data = self.validate(epoch, accum_iters, mode='val')
            
            # update the best_model
            cur_metric_value = val_log_data[self.best_metric]
            if cur_metric_value > best_metric_value:
                best_metric_value = cur_metric_value
                best_epoch = epoch
             
            self.lr_scheduler.step()

        #best_model_logger = self.val_loggers[-1]
        #assert isinstance(best_model_logger, BestModelLogger)
        #weight_path = best_model_logger.filepath()
        #if self.use_parallel:
        #    self.model.module.load(weight_path)
        #else:
        #    self.model.load(weight_path)
        #self.validate(best_epoch, accum_iter, mode='test')  # test result at best model
        #        break

        #self.logger_service.complete({
        #    'state_dict': (self._create_state_dict(epoch, accum_iter)),
        #})

    def train_one_epoch(self, epoch, accum_iters):
        average_meter_set = AverageMeterSet()
        self.model.train()
        for batch_idx, batch in enumerate(tqdm(self.train_loader)):
            batch_size = next(iter(batch.values())).size(0)
            batch = {k:v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            # append external training information to logger
            if isinstance(loss, tuple):
                loss, extra_info = loss
                for k, v in extra_info.items():
                    average_meter_set.update(k, v)
            
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()

            average_meter_set.update('loss', loss.item())
            accum_iters += batch_size

        log_data = {
            'epoch': epoch,
            'accum_iters': accum_iters,
        }
        log_data.update(average_meter_set.averages())
        self.logger.log_train(log_data)
        return accum_iters

    def validate(self, epoch, accum_iter, mode):
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        average_meter_set = AverageMeterSet()
        self.model.eval()
        with torch.no_grad():
            for batch_idx, batch in enumerate(tqdm(loader)):
                batch_size = next(iter(batch.values())).size(0)
                batch = {k:v.to(self.device) for k, v in batch.items()}

                metrics = self.calculate_metrics(batch)
                for k, v in metrics.items():
                    average_meter_set.update(k, v)
                
                #if not self.pilot:
                #    description_metrics = ['NDCG@%d' % k for k in self.metric_ks[:3]] +\
                #                          ['Recall@%d' % k for k in self.metric_ks[:3]]
                #    description = '{}: '.format(mode.capitalize()) + ', '.join(s + ' {:.3f}' for s in description_metrics)
                #    description = description.replace('NDCG', 'N').replace('Recall', 'R')
                #    description = description.format(*(average_meter_set[k].avg for k in description_metrics))
                #    tqdm_dataloader.set_description(description)

            log_data = {
                'state_dict': (self._create_state_dict(epoch, accum_iters)),
                'epoch': epoch,
                'accum_iters': accum_iters,
            }
            log_data.update(average_meter_set.averages())
            if mode == 'val':
                self.logger.log_val(log_data)
            elif mode == 'test':
                self.logger.log_test(log_data)
            else:
                raise ValueError
        return log_data
