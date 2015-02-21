import pandas as pd
import numpy as np
from os import path, listdir
from scipy import spatial
import driverfeatures as feat

def main():
    # chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks_big"
    chunk_path = r"/home/pold/Documents/Radboud/kaggle/chunks"

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
    ]

    chunks = listdir(chunk_path)

    i = 16
    for chunk in chunks:
        i += 1
        file_name = "feature_df_test_{}.h5".format(i)
        print(file_name)
        df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')
        df_features = feat.Features(df, features).extract_all_features() # extract_all_features(df, features)
        # feature_list.append(feat.Features(df, features))
        # df.append(feature_df_new, inplace = True)
        # df.reset_index(inplace = True)
        # HDF5
        df_features.to_hdf(file_name,'table')
        print("Written to", file_name)

    # feature_df_old = pd.read_hdf(r"/home/pold/Documents/Radboud/kaggle/Code/feature_df.h5", key = 'table')

if __name__ == "__main__":
    main()