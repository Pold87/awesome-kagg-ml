# coding: utf-8

import pandas as pd
import numpy as np
import pickle

from os import listdir, path

### Driver paths 
# Thomas
# drivers_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\"

# Volker

# All trips and drivers from Kaggle:
# drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers/"

# All trips from driver 1 and 100 (to save time):
drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers_small/"
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
# The lists are just for temporarily storing the data frames

list_all_drivers_all_trips = []

for driver in drivers:

    list_one_driver_all_trips = []
    trips_path = path.join(drivers_path, driver)
    trips = listdir(trips_path)

    for trip in trips:
        df = pd.read_csv(path.join(trips_path, trip))
        list_one_driver_all_trips.append(df)
        
    df_one_driver = pd.concat(list_one_driver_all_trips, axis = 0, keys = trips)
    list_all_drivers_all_trips.append(df_one_driver)

df_all_drivers = pd.concat(list_all_drivers_all_trips, axis = 0, keys = drivers)

# Pickle the dataframe
df_all_drivers.to_pickle("dataframe.pickle")



