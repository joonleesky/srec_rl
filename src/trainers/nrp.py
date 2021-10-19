from .base import BaseTrainer
from sklearn.metrics import accuracy_score, precision_score, recall_score
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class GaussianLoss(nn.Module):
    def __init__(self, reduction='none'):
        super().__init__()
        self.reduction = reduction

    def forward(self, mu, std, targets):
        var = torch.pow(std, 2)
        loss = 0.5 * (std + (mu - targets)**2 / var)
        if self.reduction == 'mean':
            return torch.mean(loss)
        elif self.reduction == 'none':
            return loss
        

class NRPTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
        # currently, NRPTrainer only supports dot_distributional head
        if self.args.head_type != 'dot_dist':
            raise ValueError
        
    @classmethod
    def code(cls):
        return 'nrp'
    
    def _create_criterion(self):
        return GaussianLoss(reduction='none')

    def calculate_loss(self, batch):
        # Get representation
        x = self.model(batch['items'], batch['ratings'], forward_head=False)
        
        # Forward Head
        B, T, H = x.shape  
        x = x.reshape(B*T, H)
        next_items = batch['next_items'].unsqueeze(-1).reshape(B*T, -1)
        mu, log_std = self.model.head.forward_dist(x, next_items)
        std = torch.exp(log_std)
        
        # Convert shape
        mu = mu.view(B*T)
        std = std.view(B*T)
        targets = batch['next_ratings'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        # Get Loss
        loss = self.criterion(mu, std, targets)
        loss = ((1-masks) * loss).mean()
        
        return loss

    def calculate_metrics(self, batch):
        # Get representation
        x = self.model(batch['items'], batch['ratings'], forward_head=False)
        
        # Forward Head
        B, T, H = x.shape  
        x = x.reshape(B*T, H)
        next_items = batch['next_items'].unsqueeze(-1).reshape(B*T, -1)
        mu, log_std = self.model.head.forward_dist(x, next_items)
        std = torch.exp(log_std)
        
        # Convert shape
        logits = mu.view(B*T)
        std = std.view(B*T)
        targets = batch['next_ratings'].view(B*T)
        masks = batch['masks'].view(B*T)
        
        # Exclude masks
        logits = torch.masked_select(logits, masks==0)
        std = torch.masked_select(std, masks==0)
        targets = torch.masked_select(targets, masks==0)
        
        # Get Metrics
        metrics = {}
        metrics['mae'] = F.l1_loss(logits, targets)
        metrics['mse'] = F.mse_loss(logits, targets)
        metrics['rmse'] = F.mse_loss(logits, targets).sqrt()
        
        metrics['mu'] = torch.mean(logits)
        metrics['std'] = torch.mean(std)
        metrics['min_std'] = torch.min(std)
        metrics['max_std'] = torch.max(std)
        metrics['nll'] = torch.mean(self.criterion(logits, std, targets))
        
        threshold = self.args.min_rating
        y_pred = (logits.cpu().numpy() >= threshold).astype(int)
        y_true = (targets.cpu().numpy() >= threshold).astype(int)
        metrics['acc'] = accuracy_score(y_pred, y_true)
        metrics['precision'] = precision_score(y_pred, y_true)
        metrics['recall'] = recall_score(y_pred, y_true)
        
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
