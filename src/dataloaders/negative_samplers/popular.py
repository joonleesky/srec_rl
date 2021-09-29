from .base import AbstractNegativeSampler
from tqdm import trange
import numpy as np
from collections import Counter


class PopularNegativeSampler(AbstractNegativeSampler):
    @classmethod
    def code(cls):
        return 'popular'

    def generate_negative_samples(self):
        np.random.seed(self.seed)
        items = np.arange(self.num_items) + 1
        popularity, total_count = self.get_popularity()
        prob = np.array([popularity[x] / total_count for x in items])
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

    def get_popularity(self):
        popularity = Counter()
        for user in range(1, self.num_users+1):
            popularity.update(self.user2dict[user]['items'])
        return popularity, sum(popularity.values())