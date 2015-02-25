# coding: utf-8

import pandas as pd
from os import listdir, path
import Helpers

########### Hic sunt dracones
#
#                               _/|__
#            _,-------,        _/ -|  \_     /~>.
#         _-~ __--~~/\ |      (  \   /  )   | / |
#      _-~__--    //   \\      \ *   * /   / | ||
#   _-~_--       //     ||      \     /   | /  /|
#  ~ ~~~~-_     //       \\     |( " )|  / | || /
#          \   //         ||    | VWV | | /  ///
#    |\     | //           \\ _/      |/ | ./|
#    | |    |// __         _-~         \// |  /
#   /  /   //_-~  ~~--_ _-~  /          |\// /
#  |  |   /-~        _-~    (     /   |/ / /
# /   /           _-~  __    |   |____|/
#|   |__         / _-~  ~-_  (_______  `\
#|      ~~--__--~ /  _     \        __\)))
# \               _-~       |     ./  \
#  ~~--__        /         /    _/     |
#        ~~--___/       _-_____/      /
#         _____/     _-_____/      _-~
#      /^<  ___       -____         -____
#         ~~   ~~--__      ``\--__       ``\
#                    ~~--\)\)\)   ~~--\)\)\)
#
##############################

def read_chunk(chunk_num, drivers_path, drivers):

    """
    Read one chunk, i.e. a subset of the drivers
    """

    print("Reading chunk number", chunk_num)

    mega_df = []
    i = 0
    for driver in drivers:

        i += 1
        print(i)
        list_one_driver_all_trips = []

        driver_fullpath = path.join(drivers_path, driver)

        trips = listdir(driver_fullpath)
    
        for trip in trips:
            trip_num = path.splitext(trip)[0]
            
            df = pd.read_csv(path.join(driver_fullpath, trip))

            # Multi-indices for driver and Trip)
            df_with_indices = pd.concat([df], keys = [(driver, trip_num)],
                                              names = ('Driver', 'Trip'))

            # We save all data frames in lists, since this avoids memory errors
            # (the lists are just for temporarily storing the data frames).
            list_one_driver_all_trips.append(df_with_indices)

        # Create dataframe from dataframe list
        df_one_driver = pd.concat(list_one_driver_all_trips)
        mega_df.append(df_one_driver)
    
    df_all_drivers = pd.concat(mega_df)

    filename = 'dataframe_' + str(chunk_num) + '.h5'
    
    # Save dataframe in HDF5
    df_all_drivers.to_hdf(path.join('chunks', filename), 'table')

    print("Written to", filename)


def read_all_chunks(drivers_path, drivers, number_of_chunks):
    """
    Reads in all drivers with all their trips and saves them
    to HDF5 files.
    """
    # Split list into parts (depending on memory capacity)
    chunked_drivers = chunks(drivers, len(drivers) // number_of_chunks)

    for chunk_num, drivers in enumerate(chunked_drivers):

        read_chunk(chunk_num, drivers_path, drivers)


def main():

    # Number of chunks (depends on memory capacities)
    number_of_chunks = 16
    
    # All trips and drivers from Kaggle:
    drivers_path = path.join("..", "drivers")
    drivers = listdir(drivers_path)
    read_all_chunks(drivers_path, drivers, number_of_chunks)


if __name__ == "__main__":
    main()






