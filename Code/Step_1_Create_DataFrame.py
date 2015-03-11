# coding: utf-8

import pandas as pd
import sys
from os import listdir, path

def read_driver(drivers_path, driver):
	print("Reading driver", driver)
	mega_df = []

	one_driver_all_trips = []

	driver_fullpath = path.join(drivers_path, driver)
	trips = listdir(driver_fullpath)

	for trip in trips:
		trip_num = path.splitext(trip)[0]
		df_trip = pd.read_csv(path.join(driver_fullpath, trip))

		# Multi-indices for driver and Trip)
		df_with_indices = pd.concat([df_trip], keys = [(driver, trip_num)],
										  names = ('Driver', 'Trip'))

		# We save all data frames in lists, since this avoids memory errors
		# (the lists are just for temporarily storing the data frames).
		one_driver_all_trips.append(df_with_indices)

	# Create dataframe from dataframe list
	df_one_driver = pd.concat(one_driver_all_trips)

	# filename = 'dataframe_' + str(chunk_num) + '.h5'
	filename = "dataframe_" + str(driver) + ".h5"

	# Save dataframe in HDF5
	df_one_driver.to_hdf(path.join('chunks', filename), 'table')
	print("Written to", filename)

def read_drivers(drivers_path, n_drivers):
	drivers = listdir(drivers_path)
	for i, driver in enumerate(drivers):
		read_driver(drivers_path, driver)

		n_drivers -= 1
		if n_drivers == 0:
			break

def main():
	# Number of chunks (depends on memory capacities)
	read_n_drivers = 2736 # 2736 is all drivers, or use -1
	if len(sys.argv) == 2:
		read_n_drivers = int(sys.argv[1])

	# All trips and drivers from Kaggle:
	drivers_path = path.join("..", "drivers")
	read_drivers(drivers_path, read_n_drivers)

if __name__ == "__main__":
	main()
