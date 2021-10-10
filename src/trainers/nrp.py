from .base import BaseTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F


class NRPTrainer(BaseTrainer):
    def __init__(self, args, model, env, train_loader, val_loader, test_loader):
        super().__init__(args, model, env, train_loader, val_loader, test_loader)
        
    @classmethod
    def code(cls):
        return 'nrp'
    
    def _create_criterion(self):
        return nn.MSELoss(reduction='none')

    def calculate_loss(self, batch):
        next_items = batch['next_items'].unsqueeze(-1)
        logits = self.model(batch['items'], batch['ratings'], next_items)
        B, T, _ = logits.shape
        
        logits = logits.view(B*T)
        targets = batch['next_ratings'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        loss = self.criterion(logits, targets)
        loss = ((1-masks) * loss).mean()
        
        return loss

    def calculate_metrics(self, batch):
        next_items = batch['next_items'].unsqueeze(-1)
        logits = self.model(batch['items'], batch['ratings'], next_items)
        B, T, _ = logits.shape
        
        logits = logits.view(B*T)
        targets = batch['next_ratings'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        logits = torch.masked_select(logits, masks==0)
        targets = torch.masked_select(targets, masks==0)
        
        metrics = {}
        metrics['mae'] = F.l1_loss(logits, targets)
        metrics['mse'] = F.mse_loss(logits, targets)
        metrics['rmse'] = F.mse_loss(logits, targets).sqrt()
        
        threshold = self.args.min_rating
        y_pred = (logits.cpu().numpy() >= threshold).astype(int)
        y_true = (targets.cpu().numpy() >= threshold).astype(int)
        metrics['acc'] = accuracy_score(y_pred, y_true)
        metrics['precision'] = precision_score(y_pred, y_true)
        metrics['recall'] = recall_score(y_pred, y_true)
        
        return metrics
