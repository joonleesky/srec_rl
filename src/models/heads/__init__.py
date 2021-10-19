from ..blocks.layers import GELU
import torch
import torch.nn as nn


class LinearPredictionHead(nn.Module):
    def __init__(self, d_model, d_out):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
            nn.Linear(d_model, d_out)
        )

    def forward(self, x, candidates=None):
        x = self.head(x)  # batch_size x d_out
        if candidates is not None:
            x = x.gather(1, candidates)  # batch_size x num_candidates
        return x


class DotProductPredictionHead(nn.Module):
    def __init__(self, embedding, d_model, d_out):
        super().__init__()
        self.embedding = embedding
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
        )
        self.bias = nn.Parameter(torch.zeros(1, d_out))

    def forward(self, x, candidates=None):
        x = self.head(x)  # B x H
        if candidates is not None: 
            emb = self.embedding(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  
            emb = self.embedding.weight[:]  
            logits = torch.matmul(x, emb.transpose(0, 1)) 
            logits += self.bias
        return logits
    

class DotProductDistributionHead(nn.Module):
    def __init__(self, embedding, d_model, d_out):
        super().__init__()
        self.embedding = embedding
        self.mu_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
        )
        self.log_std_head = nn.Sequential(
            nn.Linear(d_model, d_model),
            GELU(),
        )
        self.mu_bias = nn.Parameter(torch.zeros(1, d_out))
        self.log_std_bias = nn.Parameter(torch.zeros(1, d_out))

    def forward(self, x, candidates=None):
        x = self.mu_head(x)  # B x H
        if candidates is not None: 
            emb = self.embedding(candidates)  # B x C x H
            logits = (x.unsqueeze(1) * emb).sum(-1)  # B x C
            bias = self.mu_bias.expand(logits.size(0), -1).gather(1, candidates)  # B x C
            logits += bias
        else:  
            emb = self.embedding.weight[:]  
            logits = torch.matmul(x, emb.transpose(0, 1)) 
            logits += self.mu_bias
        return logits
    
    def forward_dist(self, x, candidates=None):
        mu = self.mu_head(x)  # B x H
        log_std = self.log_std_head(x)
        if candidates is not None: 
            emb = self.embedding(candidates)  # B x C x H
            mu = (mu.unsqueeze(1) * emb).sum(-1)  # B x C
            mu_bias = self.mu_bias.expand(mu.size(0), -1).gather(1, candidates)  # B x C
            mu += mu_bias
            
            log_std = (log_std.unsqueeze(1) * emb).sum(-1)  # B x C
            log_std_bias = self.log_std_bias.expand(log_std.size(0), -1).gather(1, candidates)  # B x C
            log_std += log_std_bias
        else:  
            emb = self.embedding.weight[:]
            mu = torch.matmul(mu, emb.transpose(0, 1)) 
            log_std = torch.matmul(log_std, emb.transpose(0, 1)) 
            mu += self.mu_bias
            log_std += self.log_std_bias

        return mu, log_std
    