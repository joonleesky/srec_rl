from .base import BaseDataset
import pandas as pd


class ML20MDataset(BaseDataset):
    @classmethod
    def code(cls):
        return 'anime'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from https://www.kaggle.com/hernan4444/anime-recommendation-database-2020/version/7')

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['rating_complete.csv']

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        file_path = folder_path.joinpath('rating_complete.csv')
        df = pd.read_csv(file_path)
        
        df['rating'] = df['rating'].progress_apply(lambda x: round(x))
        df.columns = ['uid', 'sid', 'rating', 'timestamp']
        return df