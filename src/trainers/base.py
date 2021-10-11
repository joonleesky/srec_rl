from ..common.logger import LoggerService, AverageMeterSet
import torch
import torch.nn as nn
import torch.optim as optim

from abc import *
from pathlib import Path
from tqdm import tqdm
import pandas as pd
import os


class BaseTrainer(metaclass=ABCMeta):
    def __init__(self, args, dataset, dataloader, env, model):
        self.args = args
        self.dataset = dataset
        self.train_loader, self.val_loader, self.test_loader = dataloader.get_pytorch_dataloaders()
        self.env = env
        
        self.device = args.device
        self.model = model.to(self.device)
        self.use_parallel = args.use_parallel
        if self.use_parallel:
            self.model = nn.DataParallel(self.model)
        
        self.optimizer = self._create_optimizer()
        self.lr_scheduler = self._create_scheduler()
        self.clip_grad_norm = args.clip_grad_norm
        self.criterion = self._create_criterion()

        self.logger = LoggerService(args)
        self.num_epochs = args.num_epochs
        self.metric_ks = args.metric_ks
        self.best_metric = args.best_metric
        
        self.best_metric_value = 0
        self.best_epoch = -1
        self.steps = 0

    def _create_optimizer(self):
        args = self.args
        if args.optimizer.lower() == 'adam':
            return optim.Adam(self.model.parameters(), 
                              lr=args.lr, 
                              weight_decay=args.weight_decay)
        elif args.optimizer.lower() == 'sgd':
            return optim.SGD(self.model.parameters(), 
                             lr=args.lr, weight_decay=args.weight_decay, 
                             momentum=args.momentum)
        else:
            raise ValueError
            
    def _create_scheduler(self):
        args = self.args
        return optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=args.num_epochs)
            
    @abstractmethod
    def _create_criterion(self):
        pass
    
    def _create_state_dict(self, epoch):
        return {
            'model_state_dict': self.model.module.state_dict() if self.use_parallel else self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.lr_scheduler.state_dict(),
            'epoch': epoch
        }
            
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
    
    @abstractmethod
    def recommend(self, state):
        pass

    def train(self):
        stop_training = False
        # validation at an initialization
        val_log_data = self.validate(mode='val')
        val_log_data['epoch'] = 0
        self.logger.log_val(val_log_data)
        
        for epoch in range(1, self.num_epochs+1):
            # train
            train_log_data = self.train_one_epoch()
            train_log_data['epoch'] = epoch
            self.logger.log_train(train_log_data)
            
            # validation
            val_log_data = self.validate(mode='val')
            val_log_data['epoch'] = epoch
            self.logger.log_val(val_log_data)
            
            # simulation code
            self.simulate(mode='val')
            
            
            # update the best_model
            cur_metric_value = val_log_data[self.best_metric]
            if cur_metric_value > self.best_metric_value:
                self.best_metric_value = cur_metric_value
                self.best_epoch = epoch
                best_model_state_dict = self._create_state_dict(epoch)
                self.logger.save_state_dict(best_model_state_dict)
             
            self.lr_scheduler.step()
        
        # test with the best_model
        best_model_state = self.logger.load_state_dict()['model_state_dict']
        self.model.load(best_model_state, self.use_parallel)
        test_log_data = self.validate(mode='test')
        self.logger.log_test(test_log_data)

    def train_one_epoch(self):
        average_meter_set = AverageMeterSet()
        self.model.train()
        for batch in tqdm(self.train_loader):
            batch_size = next(iter(batch.values())).size(0)
            batch = {k:v.to(self.device) for k, v in batch.items()}

            self.optimizer.zero_grad()
            loss = self.calculate_loss(batch)
            average_meter_set.update('loss', loss.item())
            if isinstance(loss, tuple):
                loss, extra_info = loss
                for k, v in extra_info.items():
                    average_meter_set.update(k, v)
            
            loss.backward()
            if self.clip_grad_norm is not None:
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            self.steps += 1
            
            # log the training information from the training process
            log_data = {'step': self.steps}
            log_data.update(average_meter_set.averages())
            self.logger.log_train(log_data)
            
        log_data = {'step': self.steps}
        log_data.update(average_meter_set.averages())
        
        return log_data

    def validate(self, mode):
        if mode == 'val':
            loader = self.val_loader
        elif mode == 'test':
            loader = self.test_loader
        else:
            raise ValueError

        average_meter_set = AverageMeterSet()
        self.model.eval()
        with torch.no_grad():
            for batch in tqdm(loader):
                batch = {k:v.to(self.device) for k, v in batch.items()}
                metrics = self.calculate_metrics(batch)
                
                num_valid_targets = torch.sum(1-batch['masks']).item()
                for k, v in metrics.items():
                    average_meter_set.update(k, v, n=num_valid_targets)

            log_data = {'step': self.steps}
            log_data.update(average_meter_set.averages())

        return log_data
    
    def simulate(self, mode):
        args = self.args
        if mode == 'val':
            user_ids = self.dataset['val_uids']
            batch_size = args.val_batch_size
        elif mode == 'test':
            user_ids = self.dataset['test_uids']
            batch_size = args.test_batch_size
        else:
            raise ValueError
        
        def chunks(lst, n):
            for i in range(0, len(lst), n):
                yield lst[i:i + n]
            
        user_ids_chunks = list(chunks(user_ids, batch_size))
        
        self.model.eval()
        with torch.no_grad():
            for batch_user_ids in tqdm(user_ids_chunks):
                max_timesteps = max(args.metric_ts)
                # cold-start simulation
                state = self.env.reset(batch_user_ids, num_interactions=args.num_cold_start)
                state = {k:v.to(self.device) for k, v in state.items()}
                item = self.recommend(state)
                
                
                
                
                # warm-start simulation
                state = self.env.reset(batch_user_ids, num_interactions=args.num_warm_start)
                
                
                import pdb
                pdb.set_trace()