from .base import BaseDataset
import pandas as pd
from collections import deque
from datetime import datetime
import time
from tqdm import tqdm
tqdm.pandas()


class NetflixDataset(BaseDataset):
    @classmethod
    def code(cls):
        return 'netflix'

    @classmethod
    def url(cls):
        raise AssertionError('download the dataset from https://www.kaggle.com/netflix-inc/netflix-prize-data')

    @classmethod
    def zip_file_content_is_folder(cls):
        return True

    @classmethod
    def all_raw_file_names(cls):
        return ['README',
                'qualifying.txt',
                'probe.txt',
                'movie_titles.csv',
                'combined_data_4.txt',
                'combined_data_3.txt',
                'combined_data_2.txt',
                'combined_data_1.txt']
    
    def _load_single_df(self, file_path):
        # Load single data-file
        df = pd.read_csv(file_path, header=None, names=['uid', 'rating', 'timestamp'], usecols=[0, 1, 2])

        # Find empty rows to slice dataframe for each movie
        tmp_movies = df[df['rating'].isna()]['uid'].reset_index()
        movie_indices = [[index, int(movie[:-1])] for index, movie in tmp_movies.values]

        # Shift the movie_indices by one to get start and endpoints of all movies
        shifted_movie_indices = deque(movie_indices)
        shifted_movie_indices.rotate(-1)

        # Gather all dataframes
        user_data = []

        # Iterate over all movies
        for [df_id_1, movie_id], [df_id_2, next_movie_id] in zip(movie_indices, shifted_movie_indices):

            # Check if it is the last movie in the file
            if df_id_1<df_id_2:
                tmp_df = df.loc[df_id_1+1:df_id_2-1].copy()
            else:
                tmp_df = df.loc[df_id_1+1:].copy()

            # Create movie_id column
            tmp_df['sid'] = movie_id

            # Append dataframe to list
            user_data.append(tmp_df)

        # Combine all dataframes
        df = pd.concat(user_data)
        del user_data, tmp_movies, tmp_df, shifted_movie_indices, movie_indices, df_id_1, movie_id, df_id_2, next_movie_id
        print('Shape:\t{}'.format(df.shape))
        return df

    def load_ratings_df(self):
        folder_path = self._get_rawdata_folder_path()
        df1 = self._load_single_df(folder_path.joinpath('combined_data_1.txt'))
        df2 = self._load_single_df(folder_path.joinpath('combined_data_2.txt'))
        df3 = self._load_single_df(folder_path.joinpath('combined_data_3.txt'))
        df4 = self._load_single_df(folder_path.joinpath('combined_data_4.txt'))
        
        df = pd.concat([df1, df2, df3, df4])
        def datetime_to_timestamp(s):
            timestamp = int(time.mktime(datetime.strptime(s, '%Y-%m-%d').timetuple()))
            return timestamp
        
        df['timestamp'] = df['timestamp'].progress_apply(lambda x: datetime_to_timestamp(x))
        df = df[['uid', 'sid', 'rating', 'timestamp']]

        return df