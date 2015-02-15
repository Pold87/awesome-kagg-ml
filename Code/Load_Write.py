# coding: utf-8

import pandas as pd
from os import listdir, path

### Driver paths 
# Thomas
# drivers_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\"

# Volker

# All trips and drivers from Kaggle:
# drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers/"

# All trips from driver 1 and 100 (to save time):
# Linux:
drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers_small"
# Windows:
# drivers_path = r"C:\Users\User\Documents\Radboud\kaggle\awesome-kagg-ml\drivers_small"

drivers = listdir(drivers_path)


# Kevin

# Arjen

# Nils

# Fran

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


# Store all trips for all drivers in a pandas data frame (multiindeces for driver and index)

mega_df = pd.DataFrame()

# Store all trips for all drivers in a pandas data frame (multiindeces for driver and index)
# The lists are just for temporarily storing the data frames
i = 0
list_all_drivers_all_trips = []

for driver in drivers:

    i += 1
    print(i)

    list_one_driver_all_trips = []
    trips_path = path.join(drivers_path, driver)
    trips = listdir(trips_path)

    for trip in trips:
        trip_num = path.splitext(trip)[0]
        
        df = pd.read_csv(path.join(trips_path, trip))
        df_with_indices = pd.concat([df], keys = [(driver, trip_num)],
                                          names = ('Driver', 'Trip'))

        list_one_driver_all_trips.append(df_with_indices)
        
    df_one_driver = pd.concat(list_one_driver_all_trips, axis = 0, keys = trips)
    list_all_drivers_all_trips.append(df_one_driver)

df_all_drivers = pd.concat(list_all_drivers_all_trips, axis = 0, keys = drivers)

# Pickle the dataframe
df_all_drivers.to_hdf('dataframe.h5','table')




