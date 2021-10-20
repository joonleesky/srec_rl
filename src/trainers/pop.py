from .base import BaseTrainer
import numpy as np
import copy


class POPTrainer(BaseTrainer):
    def __init__(self, args, dataset, dataloader, env, model):
        super().__init__(args, dataset, dataloader, env, model)
        item_cnt_dict = {}
        user_ids = self.dataset['user2dict'].keys().values
        for user_id in user_ids:
            items = self.dataset['user2dict'][user_id]['items']
            for item in items:
                if item in item_cnt_dict:
                    item_cnt_dict[item] += 1
                else:
                    item_cnt_dict[item] = 1
        
        self.pop_logits = np.zeros(self.args.num_items+1)
        for item, cnt in item_cnt_dict.items():
            self.pop_logits[item] = cnt
        
    @classmethod
    def code(cls):
        return 'pop'
    
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
        items = state['items']
        
        B, _ = items.shape
        logits = copy.deepcopy(self.pop_logits)
        logits = logits.reshape(-1,1).repeat(B, axis=1).T
        logits[np.arange(B)[:, None], items.cpu().numpy()] = 0
        logits = logits / np.sum(logits, 1, keepdims=True)
        action = []
        for logit in logits:
            action.append(np.random.choice(np.arange(len(logit)), p=logit))
        
        return np.array(action)
