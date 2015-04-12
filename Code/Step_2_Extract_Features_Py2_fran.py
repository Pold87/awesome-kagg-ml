from __future__ import division
import pandas as pd
from os import path, listdir
import Features1 as feat
import multiprocessing as mp
from numpy import random
import random
from sklearn.tree import DecisionTreeClassifier


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
           , 'speed_times_acc_std'
           , 'angle_sum'
           , 'angle_mean'
           , 'angle_acceleration_mean'
           , 'corners'
           , 'pauses_length_mean'
           , 'pauses_length_mean_rural'
           , 'pauses_length_mean_city'
           , 'pauses_length_mean_freeway'
           , 'break_distance'
           , 'radial_accel_mean'
           , 'radial_accel_median'
           , 'radial_accel_max'
           , 'radial_accel_std'            
            , 'sd_acceleration_city'
            , 'sd_acceleration_rural'
            , 'sd_acceleration_freeway'
            , 'sd_deceleration_city'
            , 'sd_deceleration_rural'
            , 'sd_deceleration_freeway'
]

# Chunks (containing parts of the mega df)
chunk_path = "/scratch/vstrobel/chunks32"
# prepend zero with rename 's/.*\_(\d{1})\..*$/dataframe_0$1.h5/' *.h5
chunks = sorted(listdir(chunk_path))


def do_job(i, chunk):
    file_name = "features_{}.h5".format(i)
    df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')


    clf = train_classifier()

    features_for_this_chunk = []

    for driver, trip in df.groupby(level = ['Driver', 'Trip']):
        # Extract all features for each trip
        series_features = feat.Features(trip, features, clf).extract_all_features()
        features_for_this_chunk.append(series_features)

    # Create a data frame with the features
    df_features_for_this_chunk = pd.DataFrame()
    df_features_for_this_chunk = df_features_for_this_chunk.from_dict(features_for_this_chunk)

    # Write data frames containing the features to HDF5 file
    df_features_for_this_chunk.to_hdf('/scratch/vstrobel/features_opti_32_fran/' + file_name, 'table')
    print("Written to", file_name)


def train_classifier():
        
    #train a decision tree classifier to distinguish between highway, rural, city
    
    #create data 
    training_data = pd.DataFrame()
    
    """number of turns
    rural, highway -> less than one per minute
    city -> one or more per minute
    """
    highway_turns = [0]*10
    rural_turns = [0]*9 + [1]
    city_turns = [3, 2, 1, 2, 0, 0, 2, 1, 1, 1]
    turn_series = pd.Series(highway_turns + rural_turns + city_turns)
    training_data.insert(0,'turns',turn_series)
    
    """duration of stops
    highway -> less than 10 seconds (stop and go) or more than 240 seconds (traffic jam), but both are rare
    rural -> between 10 seconds and 120 seconds (traffic lights), but not too common
    city -> between 10 seconds and 120 seconds (traffic lights), less than 10 seconds (stop and go)
    """
    highway_stop_duration = ([0]*8) + [5] + [480]
    rural_stop_duration = [0]*9 + [random.randint(10,120)]
    city_stop_duration = [random.randint(10,120) for i in range(8)] + [6,8]
    duration_series = pd.Series(highway_stop_duration + rural_stop_duration + city_stop_duration)
    training_data.insert(1,'duration',duration_series)
    
    """distance between stops
    highway -> less than 20meters in stop and go or traffic jam, usually jus total distance (since there are no stops)
    rural -> just total distance
    city -> between 10 and 1000 meters
    """
    highway_stop_distance = [random.randint(1000,2500) for i in range(8)] + [18,5]
    rural_stop_distance =[random.randint(500,1500) for i in range(9)] + [15]
    city_stop_distance = [random.randint(10,700) for i in range(10)]
    distance_series = pd.Series(highway_stop_distance + rural_stop_distance + city_stop_distance)
    training_data.insert(2,'distance',distance_series)
    """speed
    highway -> either below 5 mph or above 60mph
    rural -> between 40 and 65 mph
    city -> between 10 and 45 mph
    """
    highway_speed = [random.randint(60,90) for i in range(7)] + [3, 10, 110]
    rural_speed = [random.randint(40,65) for i in range(10)]
    city_speed = [random.randint(10,45) for i in range(10)]
    speed_series = pd.Series(highway_speed + rural_speed + city_speed)
    training_data.insert(3,'speed', speed_series)
    
    #train classification tree on this data

    clf = DecisionTreeClassifier(criterion='entropy')
    clf.fit(training_data, [0]*10 + [1]*10 + [2]*10)

    return clf


def main():

    jobs = []

    for i, chunk in enumerate(chunks):
        print(chunk)
        p = mp.Process(target = do_job, args = (i,chunk, ))
        jobs.append(p)
        p.start()

if __name__ == "__main__":
    main()
