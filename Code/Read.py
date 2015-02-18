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

            df = pd.read_hdf(path.join(chunk_path, chunk), key = 'table')
            for driver, trip in df.groupby(level = ['Driver', 'Trip']):

                    prob = lucky(trip)

                    # Actually write to file
                    submission_file.write("{}_{},{}\n".format(trip.index[0][0], trip.index[0][1],prob))


### Classification

def lucky(mini_df):
    return random.random()


### Features


def calc_speed_quantiles(df):
    """
    Calculate speed quantiles
    """
    diff1 = np.diff(df.x)[1:] ** 2
    diff2 = np.diff(df.y)[1:] ** 2
    s = np.sqrt(diff1 + diff2)
    return np.mquantiles(s)

def main():
    chunk_path = r"C:\Users\User\PycharmProjects\awesome-kagg-ml\chunks"
    create_submission_file(chunk_path)


if __name__ == "__main__":
    main()