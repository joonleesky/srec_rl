from .base import AbstractDataset
import pandas as pd


class TaobaoDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'taobao'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from https://tianchi.aliyun.com/dataset/dataDetail?dataId=649')

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['UserBehavior.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('UserBehavior.csv')
        df = pd.read_csv(file_path, sep=',', names=['uid','sid','category','rating','timestamp'])
        
        # convert event to rating
        df = df.replace('pv', 1)
        df = df.replace('fav', 2)
        df = df.replace('cart', 3)
        df = df.replace('buy', 4)
        df = df[['uid', 'sid', 'rating', 'timestamp']]
        return df