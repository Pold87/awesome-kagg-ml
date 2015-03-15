from __future__ import division
import pandas as pd
from os import path, listdir
import Features as feat
import multiprocessing as mp


# Feature list
features = ['trip_time'
          , 'trip_air_distance'
          , 'trip_distance'
          , 'trip_air_distance_manhattan'
          , 'median_speed'
          , 'max_speed'
          , 'max_acceleration'
          , 'max_deceleration'
          , 'median_acceleration_city'
          , 'median_acceleration_rural'
          , 'median_acceleration_freeway'
          , 'median_deceleration_city'
          , 'median_deceleration_rural'
          , 'median_deceleration_freeway'
          , 'median_acceleration'
          , 'median_deceleration'
          , 'sd_acceleration'
          , 'sd_deceleration'
          , 'sd_speed'
          , 'minimum_deceleration'
          , 'acceleration_time'
           , 'deceleration_time'
           , 'deceleration_time_city'
           , 'deceleration_time_rural'
           , 'deceleration_time_freeway'
           , 'acceleration_time_city'
           , 'acceleration_time_rural'
           , 'acceleration_time_freeway'
           , 'mean_speed_city'
           , 'mean_speed_rural'
           , 'mean_speed_sd_city'
           , 'mean_speed_sd_rural'
           , 'mean_speed_sd_freeway'
           , 'total_stop_time'
           , 'city_time_ratio'
           , 'rural_time_ratio'
           , 'freeway_time_ratio'
           , 'stop_time_ratio'
           , 'zero_acceleration_ratio_city'
           , 'zero_acceleration_ratio_rural'
           , 'zero_acceleration_ratio_freeway'
           , 'mean_speed_2'
           , 'mean_speed_3'
           , 'onicescu_energy_speed'
           , 'onicescu_energy_acc'
           , 'onicescu_energy_dec'
           , 'break_distance'
           , 'speed_times_acc_mean'
           , 'speed_times_acc_max'
           , 'speed_times_acc_min'
           , 'speed_times_acc_std'
           , 'angle_sum'
           , 'pauses_length_mean'
           , 'pauses_length_mean_rural'
           , 'pauses_length_mean_city'
]

# Chunks (containing parts of the mega df)
chunk_path = "/scratch/vstrobel/chunks32"
# prepend zero with rename 's/.*\_(\d{1})\..*$/dataframe_0$1.h5/' *.h5
chunks = sorted(listdir(chunk_path))


def do_job(i, chunk):
    file_name = "features_{}.h5".format(i)
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
    df_features_for_this_chunk.to_hdf('/scratch/vstrobel/features_opti_32/' + file_name, 'table')
    print("Written to", file_name)




def main():

    jobs = []

    for i, chunk in enumerate(chunks):
        print(chunk)
        p = mp.Process(target = do_job, args = (i,chunk, ))
        jobs.append(p)
        p.start()

if __name__ == "__main__":
    main()
