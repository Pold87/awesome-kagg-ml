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
        self.acc_and_dec = np.diff(self.euclidean_distances)
        self.accelerations = self.acc_and_dec[self.acc_and_dec > 0]
        self.decelerations = self.acc_and_dec[self.acc_and_dec < 0]
        self.angles = self.angles_helper()

    #### Helpers

    def euclidean_helper(self):
        """
        Calculate euclidean distance
        """
        return np.sqrt(self.xDiff ** 2 + self.yDiff ** 2)

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

    def median_deceleration(self):
        acc = np.median(self.decelerations)
        return acc

    def max_speed(self):
        return np.percentile(self.euclidean_distances, 90)

    def max_acceleration(self):
        return np.percentile(self.accelerations, 90)

    def max_deceleration(self):
        return np.percentile(self.decelerations, 90)
        
    def angle_sum(self):
        return self.angles.sum()
        
    def angle_mean(self):
        return self.angles.mean()

    def sd_acceleration(self):
        return np.std(self.accelerations)

    def df_deceleration(self):
        return np.std(self.decelerations)

    def sd_speed(self):
        return np.std(self.euclidean_distances)

    def minimum_deceleration(self):
        return np.percentile(self.decelerations, 10)

    def acceleration_time(self):
        return len(self.accelerations) / self.trip_time()

    def deceleration_time(self):
        return len(self.decelerations) / self.trip_time()


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
        

