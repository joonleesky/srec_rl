from .base import AbstractDataloader
import torch


class SequentialDataloader(AbstractDataloader):
    def __init__(self, args, dataset, dataset_path):
        super().__init__(args, dataset, dataset_path)

    @classmethod
    def code(cls):
        return 'sequential'

    def _get_dataset(self, mode):
        if mode == 'train':
            return self._get_train_dataset()
        elif mode == 'val':
            return self._get_eval_dataset('val')
        else:
            return self._get_eval_dataset('test')

    def _get_train_dataset(self):
        dataset = SequentialTrainDataset(self.args, 
                                         self.dataset, 
                                         self.train_negative_samples, 
                                         self.rng, 
                                         self.train_uids)
        return dataset

    def _get_eval_dataset(self, mode):
        user_ids = self.val_uids if mode=='val' else self.test_uids
        dataset = ImEvalDataset(self.args, 
                                self.dataset, 
                                self.test_negative_samples, 
                                self.rng,
                                user_ids)
        return dataset

    
class SequentialTrainDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset, negative_samples, rng, train_uids):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = train_uids
        self.train_window_size = args.train_window_size
        self.max_seq_len = args.max_seq_len
        self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        self.train_uids = train_uids

        self.index2user_and_offsets = self.populate_indices()
        self.negative_samples = negative_samples        
        self.getitem(3)
        
    def get_rng_state(self):
        return self.rng.getstate()

    def set_rng_state(self, state):
        return self.rng.setstate(state)

    def populate_indices(self):
        index2user_and_offsets = {}
        i = 0
        T = self.max_seq_len
        W = self.train_window_size

        # offset is exclusive
        # e.g. seq_len: 500, max_seq_len: 200, train_window_size: 100
        # train_indices: [0-200, 100-300, 200-400, 300-500]
        for user, pos in self.train_targets:
            if W is None or W == 0:
                offsets = [pos]
            else:
                offsets = list(range(pos, T-1, -W))  # pos ~ T
                if len(offsets) == 0:
                    offsets = [pos]
            for offset in offsets:
                index2user_and_offsets[i] = (user, offset)
                i += 1
        return index2user_and_offsets

    def __len__(self):
        return len(self.index2user_and_offsets)
    
    def getitem(self, index):
        """
        Output
            items: (S, ) sequence of interacted items
            ratings: (S, ) sequence of ratings for interacted items
            next_items: (S, ) sequence of next interacted items
            next_ratings: (S, ) sequence of ratings for next interacted items
            candidates: (S, 1+num_negatives) sequence of next items coupled with negative items
            labels: (S, ) target labels for the candidate items
        """
        user, offset = self.index2user_and_offsets[index]
        max_seq_len = self.max_seq_len
        
        item_sequence = self.user2dict[user]['items']
        rating_sequence = self.user2dict[user]['ratings']
        begin_idx = max(0, offset-max_seq_len)
        end_idx = offset+1
        item_sequence = item_sequence[begin_idx:end_idx]
        rating_sequence = rating_sequence[begin_idx:end_idx]
        
        items = item_sequence[:-1]
        ratings = rating_sequence[:-1]
        next_items = item_sequence[1:]
        next_ratings = rating_sequence[1:]

        # padding
        padding_len = max_seq_len - len(items)
        ignore_index = -100
        items = [0] * padding_len + items
        ratings = [0] * padding_len + ratings
        next_items = [0] * padding_len + next_items
        next_ratings = [0] * padding_len + next_ratings

        # initialize candidates and labels if the model is trained with negative sampling
        negatives = self.negative_samples[user]
        candidates = []
        for next_item in next_items:
            candidates.append([next_item] + negatives)
        labels = [0] * len(next_items)
        labels = [ignore_index] * padding_len + labels
        
        d = {
            'items':torch.LongTensor(items), 
            'ratings':torch.LongTensor(ratings),
            'next_items':torch.LongTensor(next_items),
            'next_ratings':torch.FloatTensor(next_ratings),
            'candidates':torch.LongTensor(candidates), 
            'labels':torch.LongTensor(labels)
        }
        
        return d


class ImEvalDataset(torch.utils.data.Dataset):
    def __init__(self, args, dataset, negative_samples, rng, eval_targets):
        self.args = args
        self.user2dict = dataset['user2dict']
        self.users = sorted(self.user2dict.keys())
        self.max_seq_len = args.max_seq_len
        self.special_tokens = dataset['special_tokens']
        self.num_users = len(dataset['umap'])
        self.num_items = len(dataset['smap'])
        self.rng = rng
        self.eval_targets = eval_targets
        
    def __len__(self):
        return len(self.eval_targets)

    def __getitem__(self, index):
        user, offset = self.eval_targets[index]
        max_seq_len = self.max_seq_len
        
        # offset is the last-target
        item_sequence = self.user2dict[user]['items']
        begin_idx = max(0, offset - max_seq_len)
        end_idx = offset
        
        # final output representation is projected onto the candidate tokens
        tokens = item_sequence[begin_idx:end_idx]
        answer = [item_sequence[end_idx]]
        negatives = self.negative_sampler[user]
        candidates = answer + negatives
        labels = [0]

        padding_len = max_len - len(tokens)
        tokens = [0] * padding_len + tokens

        # tokens: (S,)
        # labels: (1,)
        # candidates: (1+num_negatives)
        d = {
            'tokens':torch.LongTensor(tokens), 
            'candidates':torch.LongTensor(candidates), 
            'labels':torch.LongTensor(labels)
        }

        return d