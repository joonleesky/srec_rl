from .base import BaseModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn


class SAS(BaseModel):
    def __init__(self, args):
        super().__init__(args)
        num_blocks = args.num_blocks
        hidden = args.hidden_units
        num_item_ids = args.num_items + 1
        num_rating_ids = args.num_ratings + 1
        max_seq_len = args.max_seq_len
        
        self.item_embedding = nn.Embedding(num_item_ids, hidden, padding_idx=0)
        self.rating_embedding = nn.Embedding(num_rating_ids, hidden, padding_idx=0)
        self.positional_embedding = PositionalEncoding(max_seq_len, hidden)
        self.dropout = nn.Dropout(p=args.dropout)
        self.blocks = nn.ModuleList(
            [TransformerBlock(args) for _ in range(num_blocks)])
        self.norm = LayerNorm(hidden)
        self.head = self._init_head(args.head_type, hidden, num_item_ids)
        
        self.apply(NormInitializer(hidden))

    @classmethod
    def code(cls):
        return 'sas'
    
    def _init_head(self, head_type, d_model, d_out):
        if head_type == 'linear':
            head = LinearPredictionHead(hidden, d_model, d_out)
        elif head_type == 'dot':
            emb = self.item_embedding
            head = DotProductPredictionHead(emb, d_model, d_out)
        elif head_type == 'dot_dist':
            emb = self.item_embedding
            head = DotProductDistributionHead(emb, d_model, d_out)
        else:
            raise NotImplemented
        return head
    
    def forward(self, item_ids, rating_ids, candidates=None, predict=False, forward_head=True):
        # casual attention mask
        attn_mask = (item_ids > 0).unsqueeze(1).repeat(1, item_ids.size(1), 1).unsqueeze(1)
        attn_mask.tril_()
        
        # embedding
        x = self.item_embedding(item_ids) + self.rating_embedding(rating_ids)
        x = self.positional_embedding(x)
        x = self.dropout(x)
        
        # body
        for block in self.blocks:
            x = block(x, attn_mask)
        # normalization is utilized on the output with pre_LN https://tunz.kr/post/4
        x = self.norm(x)
        
        if predict:
            x = x[:, -1, :].unsqueeze(1) # (B, 1, H)
            
        # head
        if forward_head:
            B, T, H = x.shape        
            x = x.reshape(B*T, H)

            if candidates is not None:
                candidates = candidates.reshape(B*T, -1)
            x = self.head(x, candidates)
            x = x.view(B, T, -1)
        
        return x
