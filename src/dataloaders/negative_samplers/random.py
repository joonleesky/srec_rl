from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np


class RandomNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'random'

    def generate_negative_samples(self):
        assert self.seed is not None, 'Specify seed for random sampling'
        np.random.seed(self.seed)
        items = np.arange(self.num_items) + 1
        prob = np.ones_like(items)
        prob = prob / prob.sum()
        assert prob.sum() - 1e-9 <= 1.0

        negative_samples = {}
        print('Sampling negative items')
        for user in trange(1, self.num_users+1):
            seen = set(self.user2dict[user]['items'])
            
            zeros = np.array(list(seen)) - 1  # items start from 1
            p = prob.copy()
            p[zeros] = 0.0
            p = p / p.sum()
            
            num_candidates = np.sum(p>0)
            if num_candidates > self.sample_size:
                samples = np.random.choice(items, self.sample_size, replace=False, p=p)
            else:
                samples = np.random.choice(items, self.sample_size, replace=True, p=p)
            negative_samples[user] = samples.tolist()
            
        return negative_samples