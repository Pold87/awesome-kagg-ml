import pandas as pd
from os import path, listdir
import Features as feat


def main():

    # Feature list
    features = ['trip_time'
                , 'trip_air_distance'
                , 'trip_distance'
               #  , 'median_speed'
                , 'max_speed'
                , 'max_acceleration'
                , 'max_deceleration'
                , 'median_acceleration'
                , 'median_deceleration'
                , 'sd_acceleration'
                , 'df_deceleration'
                # , 'sd_speed'
                , 'minimum_deceleration'
                , 'acceleration_time'
                , 'deceleration_time'
               # , 'angle_sum'
               # , 'angle_mean'
                , 'mean_speed_city'
                , 'mean_speed_rural'
                , 'mean_speed_freeway'
                , 'mean_speed_sd_city'
                , 'mean_speed_sd_rural'
                , 'mean_speed_sd_freeway'
                , 'total_stop_time'
                , 'city_time_ratio'
                , 'rural_time_ratio'
                , 'freeway_time_ratio'
                , 'stop_time_ratio'
                , 'angle_acceeleration_mean'
                , 'angle_speed_mean'
                , 'pauses_length_mean'
                , 'pauses_length_mean_rural'
                , 'pauses_length_mean_city' # stopngo
    ]

    # Chunks (containing parts of the mega df)
    chunk_path = path.join("..", "chunks_big")
    chunks = listdir(chunk_path)

    for i, chunk in enumerate(chunks):
        print(chunk)

        file_name = "feature_df_city_accel_{}.h5".format(i)
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
        df_features_for_this_chunk.to_hdf(path.join('..', 'features', file_name), 'table')
        print("Written to", file_name)


if __name__ == "__main__":
    main()