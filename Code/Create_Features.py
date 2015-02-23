import pandas as pd
import numpy as np
from os import path, listdir
from scipy import spatial
import driverfeatures as feat
import re


def diff(a, b):
    b = set(b)
    return [aa for aa in a if aa not in b]


def main():
    chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks_big"
    # chunk_path = r"/home/pold/Documents/Radboud/kaggle/chunks"

    # Feature list
    features = [
          'trip_time'
        , 'trip_air_distance'
        , 'trip_air_distance_manhattan'
        , 'trip_distance'
        , 'average_speed'
        , 'max_speed'
        , 'average_acceleration'
        , 'average_deceleration'
        , 'average_radial_acceleration'
    ]

    chunks = listdir(chunk_path)

    i = 0
    for chunk in chunks:

        print(chunk)
        i += 1
        file_name = "feature_df_test__radial{}.h5".format(i)
        df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

        # Extract one trip at a time and not all trip

        features_for_this_chunk = []


        for driver, trip in df.groupby(level = ['Driver', 'Trip']):

            series_features = feat.Features(trip, features).extract_all_features() # extract_all_features(df, features)
            features_for_this_chunk.append(series_features)

        df_features_for_this_chunk = pd.DataFrame()
        df_features_for_this_chunk = df_features_for_this_chunk.from_dict(features_for_this_chunk)

        print(df_features_for_this_chunk.head())

        # feature_list.append(feat.Features(df, features))
        # df.append(feature_df_new, inplace = True)
        # df.reset_index(inplace = True)
        # HDF5
        df_features_for_this_chunk.to_hdf(file_name,'table')
        print("Written to", file_name)

    # feature_df_old = pd.read_hdf(r"/home/pold/Documents/Radboud/kaggle/Code/feature_df.h5", key = 'table')

if __name__ == "__main__":
    main()