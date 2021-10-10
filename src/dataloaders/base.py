from .negative_samplers import init_negative_sampler
import torch.utils.data as data_utils
from abc import *
import random


class BaseDataloader(metaclass=ABCMeta):
    def __init__(self, args, dataset, dataset_path):
        self.args = args
        seed = args.seed
        self.rng = random.Random(seed)
        self.sampler_rng = random.Random(seed) 
        save_folder = dataset_path
        self.dataset = dataset
        self.user2dict = dataset['user2dict']
        self.train_uids = dataset['train_uids']
        self.val_uids = dataset['val_uids']
        self.test_uids = dataset['test_uids']
        self.umap = dataset['umap']
        self.smap = dataset['smap']
        self.num_users = args.num_users = len(self.umap)
        self.num_items = args.num_items = len(self.smap)
        self.num_ratings = args.num_ratings = dataset['num_ratings']
        self.num_interactions = args.num_interactions = dataset['num_interactions']
        
        code = args.train_negative_sampler_code
        train_negative_sampler = init_negative_sampler(code, 
                                                       self.user2dict,
                                                       self.num_users, 
                                                       self.num_items,
                                                       args.train_negative_sample_size,
                                                       args.train_negative_sampling_seed,
                                                       save_folder)
        code = args.test_negative_sampler_code
        test_negative_sampler = init_negative_sampler(code, 
                                                      self.user2dict,
                                                      self.num_users, 
                                                      self.num_items,
                                                      args.test_negative_sample_size,
                                                      args.test_negative_sampling_seed,
                                                      save_folder)

        self.train_negative_samples = train_negative_sampler.get_negative_samples()
        self.test_negative_samples = test_negative_sampler.get_negative_samples()

    @classmethod
    @abstractmethod
    def code(cls):
        pass

    def get_pytorch_dataloaders(self):
        train_loaders = self._get_dataloaders('train')
        val_loaders = self._get_dataloaders('val')
        test_loaders = self._get_dataloaders('test')
        return train_loaders, val_loaders, test_loaders

    def _get_dataloaders(self, mode):
        batch_size = {'train':self.args.train_batch_size,
                      'val':self.args.val_batch_size,
                      'test':self.args.test_batch_size}[mode]

        dataset = self._get_dataset(mode)

        # use custom random sampler for reproducibility
        shuffle = False
        sampler = CustomRandomSampler(len(dataset), self.sampler_rng) if mode == 'train' else None
        drop_last = True if mode == 'train' else False
        dataloader = data_utils.DataLoader(dataset,
                                           batch_size=batch_size,
                                           shuffle=shuffle,
                                           sampler=sampler,
                                           pin_memory=True,
                                           num_workers=self.args.num_workers,
                                           drop_last=drop_last)
        return dataloader

    @abstractmethod
    def _get_dataset(self, mode):
        pass
    
    
class CustomRandomSampler(data_utils.Sampler):
    def __init__(self, n, rng):
        super().__init__(data_source=[]) # dummy
        self.n = n
        self.rng = rng

    def __len__(self):
        return self.n

    def __iter__(self):
        indices = list(range(self.n))
        self.rng.shuffle(indices)
        return iter(indices)

    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)