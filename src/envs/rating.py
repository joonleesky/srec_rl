from .base import BaseEnv
import torch
import copy


class RatingEnv(BaseEnv):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        
    @classmethod
    def code(cls):
        return 'rating'

    def step(self, item):
        pass
    
    def reset(self, user_ids, num_interactions): 
        item_ids = []
        rating_ids = []
        for user_id in user_ids:
            user_item_ids = self.user2dict[user_id]['items']
            user_rating_ids = self.user2dict[user_id]['ratings']
        
            if len(user_item_ids) >= num_interactions:
                item_ids.append(user_item_ids[:num_interactions])
                rating_ids.append(user_rating_ids[:num_interactions])
        
        item_ids = torch.LongTensor(item_ids)
        rating_ids = torch.LongTensor(rating_ids)
        self.state = {'items': item_ids,
                      'ratings': rating_ids}
        
        _state = copy.deepcopy(self.state)
        return _state
