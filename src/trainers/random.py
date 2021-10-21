from .base import BaseTrainer
import numpy as np


class RandomTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
        
    @classmethod
    def code(cls):
        return 'random'
    
    def _create_criterion(self):
        return None

    def calculate_loss(self, batch):
        return None

    def calculate_metrics(self, batch):
        return {}
    
    def train(self):
        test_sim_data = self.simulate(mode='test')
        self.logger.log_test(test_sim_data)

    def recommend(self, state):
        B, _ = state['items'].shape
        action = np.random.randint(low=1, 
                                   high=self.args.num_items+1, 
                                   size=B, 
                                   dtype=int)

        return action
