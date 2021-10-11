from .base import BaseEnv
import torch


class ClickEnv(BaseEnv):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        
    @classmethod
    def code(cls):
        return 'click'

    def step(self, item): 
        pass
    
    def reset(self, user_ids, num_interactions): 
        pass

