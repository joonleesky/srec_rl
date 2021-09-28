from .base import AbstractDataset
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
tqdm.pandas()


class RC15Dataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'rc15'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from https://www.kaggle.com/chadgostopp/recsys-challenge-2015')
        
    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['dataset-README.txt',
                'yoochoose-buys.dat',
                'yoochoose-clicks.dat',
                'yoochoose-test.dat']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        buy_path = folder_path.joinpath('yoochoose-buys.dat')
        click_path = folder_path.joinpath('yoochoose-clicks.dat')
        buy_df = pd.read_csv(buy_path, sep=',', names=['uid','timestamp','sid','price','quantity'])
        click_df = pd.read_csv(click_path, sep=',', names=['uid','timestamp','sid','category'])

        click_df['rating'] = 1
        buy_df['rating'] = 2
        click_df = click_df[['uid', 'sid', 'rating', 'timestamp']]
        buy_df = buy_df[['uid', 'sid', 'rating', 'timestamp']]
        df = pd.concat([click_df, buy_df])
        
        def datetime_to_timestamp(s):
            timestamp = int(time.mktime(datetime.strptime(s, '%Y-%m-%dT%H:%M:%S.%fz').timetuple()))
            return timestamp
        
        df['timestamp'] = df['timestamp'].progress_apply(lambda x: datetime_to_timestamp(x))
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df