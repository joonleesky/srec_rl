from .base import AbstractTrainer
import torch
import torch.nn as nn


class NIPTrainer(AbstractTrainer):
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
        return {'NDCG@10':0}
