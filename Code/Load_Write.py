# coding: utf-8

import pandas as pd
from os import listdir, path

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

def chunks(l, n):
    """ Yield successive n-sized chunks from l.
    """
    for i in range(0, len(l), n):
        yield l[i:i+n]


def read_chunk(chunk_num, drivers_path, drivers):

    print("Reading chunk number", chunk_num)

    list_all_drivers_all_trips = []
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
            df_with_indices = pd.concat([df], keys = [(driver, trip_num)],
                                              names = ('Driver', 'Trip'))
    
            list_one_driver_all_trips.append(df_with_indices)
            
        df_one_driver = pd.concat(list_one_driver_all_trips)
        list_all_drivers_all_trips.append(df_one_driver)
    
    df_all_drivers = pd.concat(list_all_drivers_all_trips)

    filename = 'dataframe_' + str(chunk_num) + '.h5'
    
    # Pickle the dataframe
    df_all_drivers.to_hdf(filename,'table')

    print("Written to", filename)

def read_all_chunks(drivers_path, drivers, number_of_chunks):

    # Split list in 8 parts
    chunked_drivers = chunks(drivers, len(drivers) // number_of_chunks)

    for chunk_num, drivers in enumerate(chunked_drivers):

        read_chunk(chunk_num, drivers_path, drivers)

def main():


    # Define number of chunks (depends on your PC)

    number_of_chunks = 32

    ### Driver paths 
    # Thomas
    # drivers_path = r"C:\Users\Trost\Copy\Nijmegen\Normal\Master\MasterAI\Pracitcal_ML\Data\drivers\"
    
    # Volker
    
    # All trips and drivers from Kaggle:
    drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers/"
    
    # All trips from driver 1 and 100 (to save time):
    # Linux:
    # drivers_path = r"/home/pold/Documents/Radboud/kaggle/drivers_small"
    # Windows:
    # drivers_path = r"C:\Users\User\Documents\Radboud\kaggle\awesome-kagg-ml\drivers_small"
    
    # Kevin
    
    # Arjen
    
    # Nils
    
    # Fran
    drivers = listdir(drivers_path)
    read_all_chunks(drivers_path, drivers, number_of_chunks)


if __name__ == "__main__":
    main()






