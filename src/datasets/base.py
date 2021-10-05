import os
import tempfile
import shutil
import pickle
import random

from tqdm import tqdm
from dotmap import DotMap
from abc import *
from pathlib import Path
from ..common.file_utils import *
tqdm.pandas()


class AbstractDataset(metaclass=ABCMeta):
    def __init__(self, args):
        """        
        Arguments
        ---------
        min_rating (int): minimum rating to divide positive / negative items
        min_uc (int): minimum user interactions
        min_sc (int): minimum item interactions
        eval_type (str): [leave_positive_out]
        # TODO: (append [leave_(multiple)items_out] for reinforcement learning experiments?)
        
        dataset class will be preprocessed and load by the funcion: load_dataset()
        1. download raw dataset
        2. preprocess dataset
        
        """
        self.args = args
        self.rng = random.Random(args.seed)
        self.min_rating = args.min_rating
        self.min_uc = args.min_uc
        self.min_sc = args.min_sc
        self.eval_type = args.eval_type
        self.train_ratio = args.train_ratio
        self.val_ratio = args.val_ratio
        self.test_ratio = args.test_ratio
        self.local_data_folder = args.local_data_folder

        assert self.train_ratio + self.val_ratio + self.test_ratio == 1.0
        
    @classmethod
    @abstractmethod
    def code(cls):
        pass

    @classmethod
    @abstractmethod
    def url(cls):
        pass

    @classmethod
    def is_zipfile(cls):
        return True

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return []

    @abstractmethod
    def load_ratings_df(self):
        pass

    def load_dataset(self):
        self._preprocess()
        dataset_path = self._get_preprocessed_dataset_path()
        dataset = pickle.load(dataset_path.open('rb'))
        return dataset

    def _preprocess(self):
        # (1) preprocess if necessary
        dataset_path = self._get_preprocessed_dataset_path()

        if dataset_path.is_file():
            print('Already preprocessed. Skip preprocessing')
            return
        if not dataset_path.parent.is_dir():
            dataset_path.parent.mkdir(parents=True)
        
        # (2) download if necessary
        raw_dataset_path = self._get_rawdata_folder_path()
        if raw_dataset_path.is_dir() and\
           all(raw_dataset_path.joinpath(filename).is_file() for filename in self.all_raw_file_names()):
            print('Raw data already exists. Skip downloading')
        else:
            print("Raw file doesn't exist. Downloading...")
            self._download_raw_dataset(raw_dataset_path)
        
        # (3) load dataset
        # columns: uid, sid, rating, timestamp
        df = self.load_ratings_df()

        # (4) filter triplets (min_uc, min_sc)
        df = self.filter_triplets(df)
        
        # (5) create user & item ids
        df, umap, smap = self.densify_index(df)
        
        # (6) split into train, val, test datasets
        user2dict, train_uids, val_uids, test_uids = self.split_df(df)

        special_tokens = DotMap()
        special_tokens.pad = 0
        num_ratings = len(df['rating'].unique())
        num_interactions = len(df)

        dataset = {'user2dict': user2dict,
                   'train_uids': train_uids,
                   'val_uids': val_uids,
                   'test_uids': test_uids,
                   'umap': umap,
                   'smap': smap,
                   'special_tokens': special_tokens,
                   'num_ratings': num_ratings,
                   'num_interactions': num_interactions}

        with dataset_path.open('wb') as f:
            pickle.dump(dataset, f)

    def _download_raw_dataset(self, dataset_path):
        if self.is_zipfile():
            tmproot = Path(tempfile.mkdtemp())
            tmpzip = tmproot.joinpath('file.zip')
            tmpfolder = tmproot.joinpath('folder')
            download(self.url(), tmpzip)
            unzip(tmpzip, tmpfolder)
            if self.zip_file_content_is_folder():
                tmpfolder = tmpfolder.joinpath(os.listdir(tmpfolder)[0])
            shutil.move(tmpfolder, dataset_path)
            shutil.rmtree(tmproot)
            print()
        else:
            tmproot = Path(tempfile.mkdtemp())
            tmpfile = tmproot.joinpath('file')
            download(self.url(), tmpfile)
            dataset_path.mkdir(parents=True)
            shutil.move(tmpfile, dataset_path.joinpath('ratings.csv'))
            shutil.rmtree(tmproot)
            print()

    def filter_triplets(self, df):
        print('Filtering triplets')
        if self.min_sc > 0:
            item_sizes = df.groupby('sid').size()
            good_items = item_sizes.index[item_sizes >= self.min_sc]
            df = df[df['sid'].isin(good_items)]

        if self.min_uc > 0:
            user_sizes = df.groupby('uid').size()
            good_users = user_sizes.index[user_sizes >= self.min_uc]
            df = df[df['uid'].isin(good_users)]

        return df

    def densify_index(self, df):
        print('Densifying index')
        umap = {u: (i+1) for i, u in enumerate(set(df['uid']))}
        smap = {s: (i+1) for i, s in enumerate(set(df['sid']))}
        df['uid'] = df['uid'].map(umap)
        df['sid'] = df['sid'].map(smap)
        return df, umap, smap

    def split_df(self, df):
        def sort_by_time(d):
            d = d.sort_values(by='timestamp')
            return {'items': list(d.sid), 'timestamps': list(d.timestamp), 'ratings': list(d.rating)}

        user_group = df.groupby('uid')
        user2dict = user_group.progress_apply(sort_by_time)
        """
        if self.eval_type == 'leave_positive_out':
            train_ranges = []
            val_positions = []
            test_positions = []
            for user, d in user2dict.items():
                # get the index of the last and the second last positive items
                n = len(d['items'])
                positive_items = [i for i in range(n) if d['ratings'][i] >= self.min_rating]
                train_ranges.append((user, positive_items[-2]-1))  # exclusive range
                val_positions.append((user, positive_items[-2]))
                test_positions.append((user, positive_items[-1]))
            train_targets = train_ranges
            validation_targets = val_positions
            test_targets = test_positions
        """
        if self.eval_type == 'split_users':
            uids = user2dict.keys().values
            self.rng.shuffle(uids)
            train_end_idx = int(len(uids) * self.train_ratio)
            val_end_idx = int(len(uids) * (self.train_ratio + self.val_ratio))
            
            train_uids = uids[:train_end_idx]
            val_uids = uids[train_end_idx:val_end_idx]
            test_uids = uids[val_end_idx:]
            
        else:
            raise ValueError

        return user2dict, train_uids, val_uids, test_uids

    def _get_rawdata_root_path(self):
        return Path(self.local_data_folder)

    def _get_rawdata_folder_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath(self.code())

    def _get_preprocessed_root_path(self):
        root = self._get_rawdata_root_path()
        return root.joinpath('preprocessed')

    def _get_preprocessed_folder_path(self):
        preprocessed_root = self._get_preprocessed_root_path()
        folder_name = '{}_min_rating{}-min_uc{}-min_sc{}-eval{}' \
            .format(self.code(), self.min_rating, self.min_uc, self.min_sc, self.eval_type)
        return preprocessed_root.joinpath(folder_name)

    def _get_preprocessed_dataset_path(self):
        folder = self._get_preprocessed_folder_path()
        return folder.joinpath('dataset.pkl')

