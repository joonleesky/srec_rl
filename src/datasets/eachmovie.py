from .base import AbstractDataset
import pandas as pd
from datetime import datetime
import time
from tqdm import tqdm
tqdm.pandas()


class EachMovieDataset(AbstractDataset):
    @classmethod
    def code(cls):
        return 'eachmovie'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from http://www.gatsby.ucl.ac.uk/~chuwei/data/EachMovie/eachmovie.html')

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['Vote.txt']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('Vote.txt')
        df = pd.read_csv(file_path, sep='\t', names=['uid','sid','category','rating','timestamp'])
        
        def datetime_to_timestamp(s):
            timestamp = int(time.mktime(datetime.strptime(s, '%m/%d/%y %H:%M:%S').timetuple()))
            return timestamp
        
        df['timestamp'] = df['timestamp'].progress_apply(lambda x: datetime_to_timestamp(x))
        df = df[['uid', 'sid', 'rating', 'timestamp']]

        return df