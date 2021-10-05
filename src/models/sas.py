from .base import AbstractModel
from ..common.initialization import NormInitializer
from .blocks.layers import *
from .blocks.transformer import TransformerBlock
from .heads import *
import torch
import torch.nn as nn


class SAS(AbstractModel):
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
        self.head = self._init_head(args.head_type, hidden, num_item_ids)
        self.criterion = nn.CrossEntropyLoss(reduction='none')
        
        self.apply(NormInitializer(args.model_init_range))

    @classmethod
    def code(cls):
        return 'sas'
    
    def _init_head(self, head_type, d_model, d_out):
        if head_type == 'linear':
            head = LinearPredictionHead(hidden, d_model, d_out)
        elif head_type == 'dot':
            emb = self.item_embedding
            head = DotProductPredictionHead(emb, d_model, d_out)
        else:
            raise NotImplemented
        return head
    
    def forward(self, item_ids, rating_ids, candidates):
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
        
        # head
        B, T, H = x.shape
        _, _, C = candidates.shape
        
        x = x.view(B*T, H)
        candidates = candidates.view(B*T, C)
        x = self.head(x, candidates)
        x = x.view(B, T, C)
        
        return x
        
        B, T, H = x.shape
        _, _, C = d['candidates'].shape
        x = x.view(B*T, H)
        candidates = d['candidates'].view(B*T, C)
        labels = d['labels'].view(B*T)
        masks = d['masks'].view(B*T)
        
        # masked-output
        logits = self.head(x, candidates)
        loss = self.criterion(logits, labels)
        loss = (1-masks) * loss
        loss = loss.mean()
        
        return logits.view(B, T, C), loss
        