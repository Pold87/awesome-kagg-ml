import pandas as pd
import numpy as np
from os import path, listdir
import scipy as sp
from sklearn import cluster
from sklearn.cluster import DBSCAN
from sklearn import metrics
from sklearn.datasets.samples_generator import make_blobs


def create_submission_file(df):
    """
    Create a submission file for kaggle
    """

    # Find file number for new file
    file_num = 0
    while path.isfile('submission-{}.csv'.format(file_num)):
        file_num += 1

    # Write final submission
    df.to_csv('submission-{}.csv'.format(file_num), index = False)


def extract_all_features(features, df):

    # Data frame for collecting the features
    df_features = pd.DataFrame()
   
    for feature in features:
        # Calculate value of feature
        df_features[feature.__name__] = df.groupby(level = ['Driver', 'Trip']).apply(feature)

    k_means = cluster.KMeans(n_clusters = 2, n_init = 1, n_jobs = 3)
    k_means.fit(df_features)

    # Reuse data frame for saving probabilities
    # TODO: Maybe it's better to use the Multiindex directly
    # instead of accessing it via reset_index()

    df_features.reset_index(inplace = True)

    df_features['driver_trip'] = create_first_column(df_features)
    df_features['prob'] = 1 - k_means.labels_
    return df_features.ix[:, ['driver_trip', 'prob']]

def create_first_column(df):    
    return df.ix[:, 0].apply(str) + "_" + df.ix[:, 1].apply(str)

### Features (should work on trip level and return one single number)
## TODO: Should be placed in another file

def trip_time(driver_df):
    """
    Calculate total trip time in seconds
    """    
    return len(driver_df.index)
    

def trip_air_distance(trip_df):
    """"
    Calculate air distance from starting point to end point
    """
    
    x = trip_df['x']
    y = trip_df['y']    
    
    start = [[x[0], y[0]]]
    finish = [[x[-1], y[-1]]]
    
    dist = sp.spatial.distance.cdist(start, finish, 'euclidean')
    return dist[0][0]


def trip_air_distance_manhattan(trip_df):
    """"
    Calculate air distance from starting point to end point
    """
    
    x = trip_df['x']
    y = trip_df['y']    
    
    start = [[x[0], y[0]]]
    finish = [[x[-1], y[-1]]]
    
    dist = sp.spatial.distance.cdist(start, finish, 'minkowski', 1)
    return dist[0][0]

def calc_speed(trip_df):
    """
    Calculate speed quantiles
    """
    # TODO: Think about that again
    diff1 = np.diff(trip_df.x)[1:] ** 2
    diff2 = np.diff(trip_df.y)[1:] ** 2
    s = np.sqrt(diff1 + diff2).sum()
    return s

def main():
    chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks_big"
    # chunk_path = r"/home/pold/Documents/Radboud/kaggle/chunks"
    
    # Feature list
    features = [
        trip_time
        , trip_air_distance
        , calc_speed

    ]

    chunks = listdir(chunk_path)

    final_list = []
   
    for chunk in chunks:
        print(chunk)
        df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

        # Group data frame
        for d, t in df.groupby(level = ['Driver']):
            final_list.append(extract_all_features(features, t))
        
    final_df = pd.concat(final_list)
    create_submission_file(final_df)

if __name__ == "__main__":
    main()