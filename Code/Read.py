import pandas as pd
import numpy as np
from os import path, listdir
import random

def create_submission_file(chunk_path):
    """
    Create a submission file for kaggle
    """

    # Find file number for new file
    file_num = 0

    while path.isfile("submission-{}.csv".format(file_num)):
        file_num += 1


    chunks = listdir(chunk_path)
    with open("submission-{}.csv".format(file_num), 'a') as submission_file:

        submission_file.write("driver_trip,prob\n")

        for chunk in chunks:

            print(chunk)

            df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')

            for driver, trip in df.groupby(level = ['Driver', 'Trip']):


                    # Actually write to file (TODO use pd.write_csv)
                    submission_file.write("{}_{},{}\n".format(trip.index[0][0], trip.index[0][1],prob))


### Classification

def lucky(mini_df):
    return random.random()


### Features (should work on trip level)

def trip_time(trip_df):
    """
    Calculate total trip time in seconds
    """
    return len(trip_df.index)



def trip_air_distance(trip_df):
    """"
    Calculate air distance from starting point to end point
    """


def calc_speed_quantiles(trip_df):
    """
    Calculate speed quantiles
    """
    diff1 = np.diff(trip_df.x)[1:] ** 2
    diff2 = np.diff(trip_df.y)[1:] ** 2
    s = np.sqrt(diff1 + diff2)
    return np.mquantiles(s)

def main():
    # chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks"

    chunk_path = r"/home/pold/Documents/Radboud/kaggle/chunks"
    create_submission_file(chunk_path)

if __name__ == "__main__":
    main()