import torch
import torch.nn as nn


class ItemEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        vocab_size = args.num_items + 2
        hidden = args.hidden_units
        self.emb = nn.Embedding(vocab_size, hidden, padding_idx=0)

    def forward(self, x):
        # x: B x T 
        return self.emb(x)  # B x T x H


class RatingEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        rating_size = args.num_ratings + 2
        hidden = args.hidden_units
        self.emb = nn.Embedding(1, hidden)

    def forward(self, d):
        batch_size, T = d['tokens'].shape
        return self.emb.weight.unsqueeze(0).repeat(batch_size, T, 1)  # B x T x H


class PositionalEmbedding(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.max_len
        hidden = args.hidden_units
        self.emb = nn.Embedding(max_len, hidden)

    def forward(self, d):
        x = d['tokens']
        batch_size = x.size(0)
        return self.emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B x T x H


class PositionalEmbeddingDirect(nn.Module):
    def __init__(self, args):
        super().__init__()
        max_len = args.max_len
        hidden = args.hidden_units
        self.emb = nn.Embedding(max_len, hidden)

    def forward(self, x):
        batch_size = x.size(0)
        return self.emb.weight.unsqueeze(0).repeat(batch_size, 1, 1)  # B x T x H