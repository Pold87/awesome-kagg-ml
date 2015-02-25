import pandas as pd
from os import path, listdir
import Features as feat


def main():

    # Feature list
    features = ['trip_time'
                , 'trip_air_distance'
                , 'trip_air_distance_manhattan'
                , 'trip_distance'
                , 'average_speed'
                , 'max_speed'
                , 'average_acceleration'
                , 'average_deceleration'
                , 'average_radial_acceleration'
                , 'angle'
                , 'mean_angle'
                , 'angle_sum']

    # Chunks (containing parts of the mega df)
    chunk_path = path.join("..", "chunks")
    chunks = listdir(chunk_path)

    for chunk, i in enumerate(chunks):

        file_name = "feature_df_{}.h5".format(i)
        df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

        features_for_this_chunk = []

        for driver, trip in df.groupby(level = ['Driver', 'Trip']):
            # Extract all features for each trip
            series_features = feat.Features(trip, features).extract_all_features()
            features_for_this_chunk.append(series_features)

        # Create a data frame with the features
        df_features_for_this_chunk = pd.DataFrame()
        df_features_for_this_chunk = df_features_for_this_chunk.from_dict(features_for_this_chunk)

        # Write data frames containing the features to HDF5 file
        df_features_for_this_chunk.to_hdf(file_name,'table')
        print("Written to", file_name)


if __name__ == "__main__":
    main()