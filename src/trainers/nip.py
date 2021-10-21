from .base import BaseTrainer
from ..common.metric import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn
import numpy as np


class NIPTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
        
    @classmethod
    def code(cls):
        return 'nip'
    
    def _create_criterion(self):
        return nn.CrossEntropyLoss(reduction='none')

    def calculate_loss(self, batch):
        logits = self.model(batch['items'], batch['ratings'], batch['candidates'])
        B, T, C = logits.shape
        
        logits = logits.view(B*T, -1)
        labels = batch['labels'].view(B*T)
        masks = batch['masks'].view(B*T)
        loss = self.criterion(logits, labels)
        loss = ((1-masks) * loss).mean()
        
        return loss

    def calculate_metrics(self, batch):
        logits = self.model(batch['items'], batch['ratings'], batch['candidates'])
        B, T, C = logits.shape
        
        logits = logits.view(B*T, -1)
        labels = batch['labels'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        logits = torch.masked_select(logits, (masks==0).unsqueeze(1)).view(-1, C)
        labels = torch.masked_select(labels, masks==0)
        
        metrics = recalls_and_ndcgs_for_ks(logits.cpu().numpy(), labels.cpu().numpy(), self.metric_ks)
        
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
