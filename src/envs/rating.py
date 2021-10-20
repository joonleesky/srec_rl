from .base import BaseEnv
import torch
import numpy as np
import scipy
import copy


class RatingEnv(BaseEnv):
    def __init__(self, args, dataset):
        super().__init__(args, dataset)
        
    @classmethod
    def code(cls):
        return 'rating'

    def step(self, actions):
        """
        [Params]
        actions: np.array(B,)
        
        [Returns] 
        next_state: dict({'items': np.array(B,T), 'ratings': np.array(B,T)})
        rewards: np.array(B,)
        done: bool()
        info: dict({})
        """
        _state = self._convert_state_to_tensor(self.state)
        items = _state['items']
        ratings = _state['ratings']
        actions = torch.LongTensor(actions).to(self.device)
        
        # reward model
        B, T = items.shape
        x = self.reward_model(items, ratings, forward_head=False, predict=True)
        x = x.reshape(B, -1)
        candidates = actions.unsqueeze(1)
        mus, log_stds = self.reward_model.head.forward_dist(x, candidates)
        stds = torch.exp(log_stds)
        rewards = mus.squeeze(-1)
        
        # low-confidence-bound
        confidence_level = self.args.confidence_level
        lower_bounds = []
        for mu, std in zip(mus, stds):
            lower_bounds.append(scipy.stats.norm.ppf(confidence_level, loc=mu.cpu(), scale=std.cpu()))
        
        # transition
        actions = actions.cpu().numpy()
        rewards = np.array(lower_bounds).flatten()
        for idx, (action, reward) in enumerate(zip(actions, rewards)):
            self.state['items'][idx].append(action)
            self.state['ratings'][idx].append(np.clip(round(reward), 0, self.args.num_ratings))
            self.state['rewards'][idx].append(reward)
        
        # return
        next_state = copy.deepcopy(self.state)
        self.timestep += 1
        if self.timestep >= self.max_timesteps:
            done = True
        else:
            done = False
        info = {}

        return next_state, rewards, done, info
    
    def reset(self, user_ids, num_interactions):
        """
        [Params] 
        user_ids: (list)
        num_interactions: (int)
        
        [Returns] 
        state: dict({'items': np.array(B,T), 'ratings': np.array(B,T)})
        """
        item_ids = []
        rating_ids = []
        for user_id in user_ids:
            user_item_ids = self.user2dict[user_id]['items']
            user_rating_ids = self.user2dict[user_id]['ratings']
        
            if len(user_item_ids) >= num_interactions:
                item_ids.append(user_item_ids[:num_interactions])
                rating_ids.append(user_rating_ids[:num_interactions])
        
        self.state = {'items': item_ids,
                      'ratings': rating_ids,
                      'rewards': copy.deepcopy(rating_ids)}
        self.timestep = 0
        
        # get an (estimated) number of positive items for each user
        self.state['num_positives'] = self._get_num_positives()
        
        _state = copy.deepcopy(self.state)
        return _state
    
    def _convert_state_to_tensor(self, state):
        state = {k:torch.LongTensor(v).to(self.device) for k, v in self.state.items()}
        return state
    
    def _get_num_positives(self):
        _state = self._convert_state_to_tensor(self.state)
        items = _state['items']
        ratings = _state['ratings']
        
        # compare with all the existing items
        B, T = items.shape
        x = self.reward_model(items, ratings, forward_head=False, predict=True)
        x = x.reshape(B, -1)
        mus, log_stds = self.reward_model.head.forward_dist(x)
        stds = torch.exp(log_stds)
        rewards = mus.squeeze(-1)
        
        # low-confidence-bound
        confidence_level = self.args.confidence_level
        lower_bounds = []
        for mu, std in zip(mus, stds):
            lower_bounds.append(scipy.stats.norm.ppf(confidence_level, loc=mu.cpu(), scale=std.cpu()))
        
        rewards = np.array(lower_bounds)
        
        # exclude the interacted items
        rewards[np.arange(B)[:, None], items.cpu().numpy()] = -1e9
        
        # get positive items
        num_positives = np.sum(rewards >= self.args.min_rating, 1)
        
        return num_positives
        