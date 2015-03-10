import pandas as pd
import numpy as np
from scipy import spatial


class Features:
    def __init__(self, df, features):
        self.df = df
        self.features = features
        self.driver = df.index.get_level_values('Driver')[0]
        self.trip = df.index.get_level_values('Trip')[0]
        self.xDiff = np.diff(self.df.x)
        self.yDiff = np.diff(self.df.y)
        self.x_start = df['x'][0]
        self.y_start = df['y'][0]
        self.y_start = df['y'][0]
        self.x_finish = df['x'][-1]
        self.y_finish = df['y'][-1]
        self.euclidean_distances = self.euclidean_helper()
        self.total_time = self.trip_time()
        self.acc_and_dec = np.diff(self.euclidean_distances)
        self.accelerations = self.acc_and_dec[self.acc_and_dec > 0]
        self.decelerations = self.acc_and_dec[self.acc_and_dec < 0]
        self.city_mask = (self.euclidean_distances > 5) & (self.euclidean_distances < 45)
        self.city_speeds = self.euclidean_distances[self.city_mask]
        self.rural_mask = (self.euclidean_distances > 40) & (self.euclidean_distances < 60)
        self.rural_speeds = self.euclidean_distances[self.rural_mask]
        self.freeway_mask = (self.euclidean_distances > 55) & (self.euclidean_distances < 120)
        self.freeway_speeds = self.euclidean_distances[self.freeway_mask]
        self.city_acc_and_dec = self.acc_and_dec[self.city_mask[:-1]]
        self.rural_acc_and_dec = self.acc_and_dec[self.rural_mask[:-1]]
        self.freeway_acc_and_dec = self.acc_and_dec[self.freeway_mask[:-1]]
        # self.angles = np.array(self.angles_helper())
        self.stop_time = self.total_stop_time()
        self.pauses = self.pauses_helper()
        self.rural_acc_mask = (self.rural_acc_and_dec > 0.5)
        self.rural_dec_mask = (self.rural_acc_and_dec < 0.5)
        self.rural_accs = self.rural_acc_and_dec[self.rural_acc_mask]
        self.rural_decs = self.rural_acc_and_dec[self.rural_dec_mask]

        self.city_acc_mask = (self.city_acc_and_dec > 0.5)
        self.city_dec_mask = (self.city_acc_and_dec < 0.5)
        self.city_accs = self.city_acc_and_dec[self.city_acc_mask]
        self.city_decs = self.city_acc_and_dec[self.city_dec_mask]

        self.freeway_acc_mask = (self.freeway_acc_and_dec > 0.5)
        self.freeway_dec_mask = (self.freeway_acc_and_dec < 0.5)
        self.freeway_accs = self.freeway_acc_and_dec[self.freeway_acc_mask]
        self.freeway_decs = self.freeway_acc_and_dec[self.freeway_dec_mask]


    #### Helpers

    def euclidean_helper(self):
        """
        Calculate euclidean distance
        """
        # Calculate miles per hour (I assume it's somewhere in the US)
        return np.sqrt(self.xDiff ** 2 + self.yDiff ** 2) * 2.2369

    def euclidean_helper_2(self):
        """
        Calculate euclidean distance between point t and point t+2
        """
        # TODO: Think about that again
        diff1 = np.subtract(self.df.x[3:], self.df.x[1:-2]) ** 2
        diff2 = np.subtract(self.df.y[3:], self.df.y[1:-2]) ** 2
        return np.sqrt(diff1 + diff2)

    def angles_helper(self):
        return np.degrees(np.arctan2(np.diff(self.df.y), np.diff(self.df.x)))
        
    ### NEW
    def acceleration_mask(self):
        return (self.acc_and_dec > 0)
    
    ###NEW
    # not sure if this works
    def pauses_helper(self):
        """ create bool array that is true if car moves"""
        return np.array(self.euclidean_distances > 0) 

    ### Features

    def trip_time(self):
        """
        Calculate total trip time in seconds
        """
        return len(self.df.index)

    def trip_air_distance(self):
        """"
        Calculate air distance from starting point to end point
        """
        start = [[self.x_start, self.y_start]]
        finish = [[self.x_finish, self.y_finish]]

        dist = spatial.distance.cdist(start, finish, 'euclidean')
        return dist[0][0]

    def trip_air_distance_manhattan(self):
        """"
        Calculate air distance from starting point to end point
        """

        start = [[self.x_start, self.y_start]]
        finish = [[self.x_finish, self.y_finish]]

        dist = spatial.distance.cdist(start, finish, 'minkowski', 1)
        return dist[0][0]

    def trip_distance(self):
        """
        Calculate speed quantiles
        """
        return self.euclidean_distances.sum()

    def median_speed(self):
        """
        Calculate speed quantiles
        """
        return np.median(self.euclidean_distances)

    def median_acceleration(self):
        acc = np.median(self.accelerations)
        return acc

    def median_acceleration_city(self):

        return self.zero_or_mean(self.city_accs)

    def median_acceleration_rural(self):

        return self.zero_or_mean(self.rural_accs)

    def median_acceleration_freeway(self):
        return self.zero_or_mean(self.freeway_accs)

    def median_deceleration_city(self):
        return self.zero_or_mean(self.city_decs)

    def median_deceleration_rural(self):
        return self.zero_or_mean(self.rural_decs)

    def median_deceleration_freeway(self):
        return self.zero_or_mean(self.freeway_decs)

    def median_deceleration(self):
        acc = np.median(self.decelerations)
        return acc

    def max_speed(self):
        # Could be done differently now
        return np.percentile(self.euclidean_distances, 90)

    def max_acceleration(self):
        return np.percentile(self.accelerations, 90)

    def max_deceleration(self):
        return np.percentile(self.decelerations, 90)
        
    def angle_sum(self):
        return self.angles.sum()
        
    def angle_mean(self):
        return self.angles.mean()
    
    #####New
    def angle_acceeleration_mean(self):
        ##ToDo: match sizes of arrays
        return np.mean(self.angles[self.acceleration_mask()]/self.accelerations)
    ####New
    def angle_speed_mean(self):
        return np.mean(self.angles/self.euclidean_distances)
        
    ####NEW 
    # I think it works, but I haven't tested it.
    def pauses_length_mean(self):
        return self.zero_or_mean(self.pauses) #self.pauses.mean()
    
    ####NEW
    # works on toy problems, under the assumption that city_mask is a numpy 
    #bool array
    def pauses_length_mean_rural(self):
        return self.zero_or_mean(self.pauses[-self.city_mask])
    
    #### NEW
    def pauses_length_mean_city(self):
        return self.zero_or_mean(self.pauses[self.city_mask])
       

    def sd_acceleration(self):
        return np.std(self.accelerations)

    def sd_deceleration(self):
        return np.std(self.decelerations)

    def sd_speed(self):
        return np.std(self.euclidean_distances)

    def minimum_deceleration(self):
        return np.percentile(self.decelerations, 10)

    def acceleration_time(self):
        return len(self.accelerations) / self.total_time


    def acceleration_time_city(self):

        return len(self.city_accs) / self.total_time

    def acceleration_time_rural(self):
        return len(self.rural_accs) / self.total_time

    def acceleration_time_freeway(self):

        return len(self.freeway_accs) / self.total_time

    def deceleration_time_city(self):

        return len(self.city_decs) / self.total_time

    def deceleration_time_rural(self):

        return len(self.rural_decs) / self.total_time

    def deceleration_time_freeway(self):

        return len(self.freeway_decs) / self.total_time

    def deceleration_time(self):
        return len(self.decelerations) / self.total_time

    def zero_or_mean(self, speeds):

        if len(speeds) == 0:
            return 0
        else:
            return np.mean(speeds)

    def mean_speed_city(self):

        return self.zero_or_mean(self.city_speeds)

    def mean_speed_rural(self):

        return self.zero_or_mean(self.rural_speeds)

    def mean_speed_freeway(self):

        return self.zero_or_mean(self.freeway_speeds)

    def zero_or_std(self, speeds):

        if len(speeds) == 0:
            return 0
        else:
            return np.std(speeds)

    def mean_speed_sd_city(self):

        return self.zero_or_std(self.city_speeds)

    def mean_speed_sd_rural(self):
        return self.zero_or_std(self.rural_speeds)

    def mean_speed_sd_freeway(self):
        return self.zero_or_std(self.freeway_speeds)

    def city_time_ratio(self):
        return len(self.city_speeds) / self.total_time

    def rural_time_ratio(self):
        return len(self.rural_speeds) / self.total_time

    def freeway_time_ratio(self):

        time = self.freeway_speeds

        return len(time) / self.total_time

    def total_stop_time(self):
        mask = self.euclidean_distances < 5
        return len(self.euclidean_distances[mask])

    def stop_time_ratio(self):
        return self.stop_time / self.total_time

    def zero_acceleration_ratio_city(self):

        zero_acc_mask = (self.city_acc_and_dec > - 0.5) & (self.city_acc_and_dec < 0.5)

        return len(self.city_acc_and_dec[zero_acc_mask]) / self.total_time

    def zero_acceleration_ratio_rural(self):

        zero_acc_mask = (self.rural_acc_and_dec > -0.5) & (self.rural_acc_and_dec < 0.5)

        return len(self.rural_acc_and_dec[zero_acc_mask]) / self.total_time

    def zero_acceleration_ratio_freeway(self):

        zero_acc_mask = (self.freeway_acc_and_dec > -0.5) & (self.freeway_acc_and_dec < 0.5)

        return len(self.freeway_acc_and_dec[zero_acc_mask]) / self.total_time

    def extract_all_features(self):
        # Data frame for collecting the features
        series_features = pd.Series()
        series_features['Driver'] = self.driver
        series_features['Trip'] = self.trip

        for feature in self.features:
            feature_method = getattr(self, feature)
            # Calculate value of feature
            series_features[feature] = feature_method()

        return series_features
        

