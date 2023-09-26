import os
import chardet
import pandas as pd
from surprise import Dataset, Reader


def load_data(data_path):
    '''

    :param cols:
    :param dataset:
    :param data_path:
    :return:
    '''

    data = pd.read_parquet(
        data_path,
    )

    print(data.columns)
    return data


def reader_data(data, cols, model=None, scale=False):
    '''
    :param scale:
    :param cols:
    :param data:
    :param model:
    :return:
    '''

    if model == 'surprise':
        if scale:
            reader = Reader(rating_scale=(1, 5))
            data = Dataset.load_from_df(data[cols],
                                        reader)

        return data

if __name__ == "__main__":
    data = load_data(data_path="/home/abilfad/Documents/dicoding/project/dicoding-RecSys/data/dicoding_user_item_rating.gzip")
    print(data.head())
    user_item = reader_data(data, cols = data.columns)


