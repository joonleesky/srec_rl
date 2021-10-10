from .base import BaseDataset
import pandas as pd


class RetailRocketDataset(BaseDataset):
    @classmethod
    def code(cls):
        return 'retailrocket'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from https://www.kaggle.com/retailrocket/ecommerce-dataset')

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['category_tree.csv',
                'events.csv',
                'item_properties_part1.csv',
                'item_properties_part2.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('events.csv')
        df = pd.read_csv(file_path, sep=',')
        
        # convert event to rating
        df = df.replace('view', 1)
        df = df.replace('addtocart', 2)
        df = df.replace('transaction', 3)
        df = df.rename({'visitorid': 'uid', 
                        'itemid': 'sid',
                        'event': 'rating'}, axis='columns')
        df = df[['uid', 'sid', 'rating', 'timestamp']]
        return df