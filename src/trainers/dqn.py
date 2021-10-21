from .base import BaseTrainer
from ..common.metric import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import numpy as np
import copy


class DQNTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
        self.target_model = copy.deepcopy(model)
        self.update_step = 0
        
    @classmethod
    def code(cls):
        return 'dqn'
    
    def _create_criterion(self):
        return nn.SmoothL1Loss(reduction='none')
    
    def _update_target_model(self, main, target):
        pass

    def calculate_loss(self, batch):
        # get current and target q-value
        B, T = batch['items'].shape
        q_value = self.model(batch['items'], batch['ratings'], batch['next_items'])
        reward = batch['next_ratings']
        with torch.no_grad():
            target_q_value = self.target_model(batch['next_items'], batch['next_ratings'].long())
            target_q_value = torch.max(target_q_value, -1)[0]
        
        q_value = q_value.flatten()
        reward = reward.flatten()
        target_q_value = target_q_value.flatten()
        target = reward + self.args.gamma * target_q_value
        
        masks = batch['masks'].flatten()
        loss = self.criterion(q_value, target)
        loss = ((1-masks) * loss).mean()
        
        self.update_step += 1
        if self.update_step % self.args.target_update_freq == 0:
            self.target_model = copy.deepcopy(self.model)
        
        return loss

    def calculate_metrics(self, batch):
        metrics = {}
        loss = self.calculate_loss(batch)
        metrics['loss'] = loss.cpu().item()
        
        return metrics

    def recommend(self, state):
        items = state['items']
        ratings = state['ratings']
        
        B, T = state['items'].shape
        logits = self.model(items, ratings, predict=True).squeeze(1)

        # interacted items must be excluded
        logits = logits.cpu().numpy()
        logits[np.arange(B)[:, None], items.cpu().numpy()] = -1e9
        action = np.argmax(logits, 1)

        return action
