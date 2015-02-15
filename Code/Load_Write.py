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
drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers_small/"
# Windows:
drivers_path = r"C:\Users\User\Documents\Radboud\kaggle\awesome-kagg-ml\drivers_small"

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

for driver in drivers:

    trips_path = path.join(drivers_path, driver)
    trips = listdir(trips_path)

    for trip in trips:

        # Get file name without extension        
        trip_num = path.splitext(trip)[0]
        
        # Read in trip
        df = pd.read_csv(path.join(trips_path, trip))
        
        # Add trip to mega data frame
        mega_df = pd.concat([mega_df, 
                                    pd.concat([df], keys = [(driver, trip_num)])])
    
# Save dataframe in HDF5 format
mega_df.to_hdf('dataframe.h5','table')



