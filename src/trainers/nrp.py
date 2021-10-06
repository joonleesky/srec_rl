from .base import AbstractTrainer
from ..common.metric import recalls_and_ndcgs_for_ks
import torch
import torch.nn as nn


class NRPTrainer(AbstractTrainer):
    def __init__(self, args, model, train_loader, val_loader, test_loader):
        super().__init__(args, model, train_loader, val_loader, test_loader)
        
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
        
        metrics = recalls_and_ndcgs_for_ks(logits, labels, self.metric_ks)
        
        return metrics
